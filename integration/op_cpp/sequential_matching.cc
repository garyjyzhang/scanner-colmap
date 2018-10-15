#include "colmap.pb.h"
#include "io.cc"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"
#include "scanner/util/memory.h"

#include <colmap/base/database.h>
#include <colmap/feature/matching.h>
#include <colmap/feature/sift.h>
#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>

using colmap::camera_t;
using colmap::FeatureDescriptors;
using colmap::FeatureKeypoint;
using colmap::FeatureKeypoints;
using colmap::FeatureMatches;
using colmap::Image;
using colmap::image_pair_t;
using colmap::image_t;
using colmap::TwoViewGeometry;

using scanner::u8;
// Kernel for Colmap sequential feature matching.
// Expect a stencil of images.
// Matches the center image to all other images in the window
//
// +-------------------------------+-----------------------> images[i]
//                      ^          |           ^
//                      |   Current image[i]   |
//                      |          |           |
//                      +----------+-----------+
//                                 |
//                        Match image_i against
//
//                    image_[i - o, i + o]        with o = [1 .. overlap]
//
// config:
class SequentialMatchingCPUKernel : public scanner::StenciledBatchedKernel,
                                    public scanner::VideoKernel {
public:
  SequentialMatchingCPUKernel(const scanner::KernelConfig &config)
      : scanner::StenciledBatchedKernel(config) {
    parseConfigs(config);
  }

  // create SIFT and sequential matching options
  void parseConfigs(const scanner::KernelConfig &config) {
    SequentialMatchingArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());

    auto &siftArgs = args.siftargs();
    // set sift_options_
    sift_options_.use_gpu = siftArgs.use_gpu();
    sift_options_.gpu_index = siftArgs.gpu_index();
    sift_options_.max_ratio = siftArgs.max_ratio();
    sift_options_.max_distance = siftArgs.max_distance();
    sift_options_.cross_check = siftArgs.cross_check();
    sift_options_.max_num_matches = siftArgs.max_num_matches();
    sift_options_.max_error = siftArgs.max_error();
    sift_options_.confidence = siftArgs.confidence();
    sift_options_.min_num_trials = siftArgs.min_num_trials();
    sift_options_.max_num_trials = siftArgs.max_num_trials();
    sift_options_.min_inlier_ratio = siftArgs.min_inlier_ratio();
    sift_options_.min_num_inliers = siftArgs.min_num_inliers();
    sift_options_.multiple_models = siftArgs.multiple_models();
    sift_options_.guided_matching = siftArgs.guided_matching();

    // set sequential matching args
    sequential_matching_options_.loop_detection = args.loop_detection();
    sequential_matching_options_.overlap = args.overlap();
    sequential_matching_options_.quadratic_overlap = args.quadratic_overlap();

    // set two view geometry verify args, see colmap/feature/matching.cc
    // TwoViewGeometryVerifier
    two_view_geometry_options_.min_num_inliers =
        static_cast<size_t>(sift_options_.min_num_inliers);
    two_view_geometry_options_.ransac_options.max_error =
        sift_options_.max_error;
    two_view_geometry_options_.ransac_options.confidence =
        sift_options_.confidence;
    two_view_geometry_options_.ransac_options.min_num_trials =
        static_cast<size_t>(sift_options_.min_num_trials);
    two_view_geometry_options_.ransac_options.max_num_trials =
        static_cast<size_t>(sift_options_.max_num_trials);
    two_view_geometry_options_.ransac_options.min_inlier_ratio =
        sift_options_.min_inlier_ratio;
  }

  void printKeypoint(colmap::FeatureKeypoint &point) {
    printf("point: %f %f %f %f %f %f \n", point.x, point.y, point.a11,
           point.a12, point.a21, point.a22);
  }

  // converted from colmap::TwoViewGeometryVerifier::Run
  void verifyTwoViewGeometry(TwoViewGeometry &two_view_geometry,
                             FeatureMatches &matches,
                             FeatureKeypoints &keypoints1,
                             FeatureKeypoints &keypoints2) {
    // use dummy camera for now
    colmap::Camera camera1, camera2;

    const auto points1 = colmap::FeatureKeypointsToPointsVector(keypoints1);
    const auto points2 = colmap::FeatureKeypointsToPointsVector(keypoints2);

    if (sift_options_.multiple_models) {
      two_view_geometry.EstimateMultiple(camera1, points1, camera2, points2,
                                         matches, two_view_geometry_options_);
    } else {
      two_view_geometry.Estimate(camera1, points1, camera2, points2, matches,
                                 two_view_geometry_options_);
    }
  }

  void execute(const scanner::StenciledBatchedElements &input_cols,
               scanner::BatchedElements &output_cols) override {

    auto &image_stencil = input_cols[0][0];
    auto &keypoint_stencil = input_cols[1][0];
    auto &descriptor_stencil = input_cols[2][0];

    std::vector<Image> images;
    std::vector<FeatureKeypoints> keypoints_list;
    std::vector<FeatureDescriptors> descriptors_list;

    // read in images, descriptors, and keypoints
    for (int i = 0; i < image_stencil.size(); i++) {
      std::cout << "read image" << std::endl;
      images.push_back(readSingleFromElement<Image>(image_stencil[i]));
      std::cout << "read keypoints" << std::endl;
      keypoints_list.push_back(
          readVectorFromElement<FeatureKeypoints>(keypoint_stencil[i]));
      std::cout << "read descriptors" << std::endl;
      descriptors_list.push_back(
          ReadMatrixFromElement<FeatureDescriptors>(descriptor_stencil[i]));

      std::cout << "Read image id: " << images[i].ImageId() << std::endl;
    }
    std::cout << "stencil size: " << images.size() << std::endl;

    // left-most image to match with every other image
    Image image1 = images[0];
    image_t image_id1 = image1.ImageId();
    FeatureDescriptors &descriptors1 = descriptors_list[0];
    FeatureKeypoints &keypoints1 = keypoints_list[0];

    MatchingResult matching_result(image1);

    for (int i = 1; i < images.size(); i++) {
      Image image2 = images[i];
      image_t image_id2 = image2.ImageId();
      FeatureDescriptors descriptors2 = descriptors_list[i];
      FeatureKeypoints keypoints2 = keypoints_list[i];
      matching_result.add_peer(image2);

      std::cout << "matching image " << image1.ImageId() << " with image "
                << image2.ImageId() << std::endl;

      // Create pair id
      image_pair_t pair_id =
          colmap::Database::ImagePairToPairId(image_id1, image_id2);

      // Run matching
      FeatureMatches featureMatches;
      colmap::MatchSiftFeaturesCPU(sift_options_, descriptors1, descriptors2,
                                   &featureMatches);

      TwoViewGeometry two_view_geometry;
      // Run Two View Geometry Verification
      verifyTwoViewGeometry(two_view_geometry, featureMatches, keypoints1,
                            keypoints2);

      // TODO: add guided matching

      if (featureMatches.size() <
          static_cast<size_t>(sift_options_.min_num_inliers)) {
        printf("FeatureMatches for %d and %d has too few inliers", image_id1,
               image_id2);
        featureMatches.clear();
      }

      if (two_view_geometry.inlier_matches.size() <
          static_cast<size_t>(sift_options_.min_num_inliers)) {
        printf("View geometry for %d and %d has too few inliers", image_id1,
               image_id2);
        two_view_geometry = TwoViewGeometry();
      }

      matching_result.add_feature_matches(featureMatches);
      matching_result.add_two_view_geometry(two_view_geometry);

      // write matching result to output column
      std::cout << "inserting to columns..." << std::endl;

      writeMatchingResultToColumn(output_cols[0], matching_result);
    }
  }

private:
  colmap::SiftMatchingOptions sift_options_;
  colmap::SequentialMatchingOptions sequential_matching_options_;
  colmap::TwoViewGeometry::Options two_view_geometry_options_;
};

REGISTER_OP(SequentialMatchingCPU)
    .stencil()
    .input("image_ids")
    .input("keypoints")
    .input("descriptors")
    .output("matching_results")
    .protobuf_name("featureMatchingArgs");

REGISTER_KERNEL(SequentialMatchingCPU, SequentialMatchingCPUKernel)
    .device(scanner::DeviceType::CPU)
    .batch()
    .num_devices(1);
