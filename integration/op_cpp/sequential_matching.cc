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

    auto &image_id_stencil = input_cols[0][0];
    auto &keypoint_stencil = input_cols[1][0];
    auto &descriptor_stencil = input_cols[2][0];

    std::vector<image_t> image_ids;
    std::vector<FeatureKeypoints> keypoints;
    std::vector<FeatureDescriptors> descriptors;

    for (int i = 0; i < image_id_stencil.size(); i++) {
      image_ids.push_back(
          *reinterpret_cast<image_t *>(image_id_stencil[i].buffer));

      std::cout << "image id: " << image_ids[i] << std::endl;
      // Read descriptors
      descriptors.push_back(ReadDescriptorsFromElement(descriptor_stencil[i]));

      keypoints.push_back(readKeypointsFromElement(keypoint_stencil[i]));
    }
    std::cout << "stencil size: " << image_ids.size() << std::endl;

    int center_idx = descriptors.size() / 2;

    for (int i = 0; i < image_ids.size(); i++) {
      if (i == center_idx)
        continue;

      image_t image_id1 = image_ids[center_idx];
      image_t image_id2 = image_ids[i];
      FeatureDescriptors descriptors1 = descriptors[center_idx];
      FeatureDescriptors descriptors2 = descriptors[i];
      FeatureKeypoints keypoints1 = keypoints[center_idx];
      FeatureKeypoints keypoints2 = keypoints[i];

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
        featureMatches.clear();
      }

      if (two_view_geometry.inlier_matches.size() <
          static_cast<size_t>(sift_options_.min_num_inliers)) {
        two_view_geometry = TwoViewGeometry();
      }

      std::cout << "inserting to columns..." << std::endl;

      writeSingleToColumn<image_pair_t>(output_cols[0], pair_id);
      writeFeatureMatchesToColumn(output_cols[1], featureMatches);
      writeSingleToColumn<TwoViewGeometry>(output_cols[2], two_view_geometry);
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
    .output("pair_id")
    .output("feature_matches")
    .output("two_view_geometry")
    .protobuf_name("featureMatchingArgs");

REGISTER_KERNEL(SequentialMatchingCPU, SequentialMatchingCPUKernel)
    .device(scanner::DeviceType::CPU)
    .num_devices(1);
