#include "mapping_cluster_cache.h"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"
#include "scanner/util/memory.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

// converted from DatabaseCache::Load()
void MappingClusterCache::LoadFromStencils(
    const Elements &image_id_stencil, const Elements &pair_image_id_stencil,
    const Elements &two_view_geometry_stencil,
    const Elements &keypoints_stencil, const Elements &camera_stencil,
    int num_pivot_images, const size_t min_num_matches,
    const bool ignore_watermarks) {

  ////////////////////////////////////////////////////////////////////////////
  // load all the resources from stencils
  ////////////////////////////////////////////////////////////////////////////
  std::cout << "Loading cache" << std::endl;
  int stencil_size = image_id_stencil.size();

  vector<image_t> pivot_image_ids;

  for (int i = 0; i < image_id_stencil.size(); i++) {
    std::cout << "start reading image id" << std::endl;
    // load image id
    image_t image_id = readSingleFromElement<image_t>(image_id_stencil[i]);

    // the first num_pivot_images are pivots
    if (i < num_pivot_images) {
      pivot_image_ids.push_back(image_id);
    }

    std::cout << "Read image id: " << image_id << std::endl;

    // load camera
    colmap::Camera camera;
    readCameraFromElement(camera_stencil[i], &camera);
    std::cout << "Loaded camera " << camera.CameraId() << std::endl;
    std::cout << "prior focal length" << camera.HasPriorFocalLength()
              << std::endl;
    cameras.emplace(camera.CameraId(), camera);

    // load keypoint
    auto keypoints = readKeypointsFromElement(keypoints_stencil[i]);

    // create image
    images.emplace(image_id,
                   createImage(image_id, camera.CameraId(), keypoints));
    std::cout << "Loaded image id: " << image_id << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////
  // Build correspondence graph
  ////////////////////////////////////////////////////////////////////////////

  auto UseInlierMatchesCheck = [min_num_matches, ignore_watermarks](
                                   const TwoViewGeometry &two_view_geometry) {
    return static_cast<size_t>(two_view_geometry.inlier_matches.size()) >=
               min_num_matches &&
           (!ignore_watermarks ||
            two_view_geometry.config != TwoViewGeometry::WATERMARK);
  };

  std::cout << "building correspondence graph" << std::endl;
  vector<PairImageIds> pairs;
  vector<TwoViewGeometries> tvg_list;
  tvg_list.reserve(num_pivot_images);
  pairs.reserve(num_pivot_images);

  // only iterate over the pivot images
  for (int i = 0; i < num_pivot_images; i++) {
    image_t pivot_image_id = pivot_image_ids[i];
    std::cout << "unpacking matches for pivot image: " << pivot_image_id
              << std::endl;

    // load pairs
    vector<image_t> pair_image_ids =
        readVectorFromElement<PairImageIds>(pair_image_id_stencil[i]);
    printf("Read %ld pair ids", pairs[i].size());

    // load two view geometries
    vector<TwoViewGeometry> two_view_geometries =
        readTwoViewGeometries(two_view_geometry_stencil[i]);
    printf("Read %ld tvgs", tvg_list[i].size());
    for (TwoViewGeometry tvg : two_view_geometries) {
      for (auto &featureMatch : tvg.inlier_matches) {
        std::cout << "feature match: " << featureMatch.point2D_idx1 << " "
                  << featureMatch.point2D_idx2 << std::endl;
      }
    }

    // number of pairs must be equal to number of tvgs, since they are 1-to-1
    assert(pair_image_ids.size() == two_view_geometries.size());

    for (int j = 0; j < pair_image_ids.size(); j++) {
      image_t pair_image_id = pair_image_ids[j];
      if (UseInlierMatchesCheck(two_view_geometries[j])) {
        if (ExistsImage(pair_image_id)) {
          correspondence_graph.AddCorrespondences(
              pivot_image_id, pair_image_id,
              two_view_geometries[j].inlier_matches);
        } else {
          printf("Ignoring pair %d %d because latter is not found in cache",
                 pivot_image_id, pair_image_id);
        }
      } else {
        printf("Ignoring pair %d %d because inlier check failed",
               pivot_image_id, pair_image_id);
      }
    }
  }

  correspondence_graph.Finalize();
  // Set number of observations and correspondences per image.
  for (auto &image : images) {
    image.second.SetNumObservations(
        correspondence_graph.NumObservationsForImage(image.first));
    image.second.SetNumCorrespondences(
        correspondence_graph.NumCorrespondencesForImage(image.first));
  }

  std::cout << "cache loading finished" << std::endl;
}

colmap::Image MappingClusterCache::createImage(image_t id,
                                               colmap::camera_t camera_id,
                                               FeatureKeypoints &keypoints) {
  colmap::Image image;
  image.SetName("image_" + std::to_string(id));
  image.SetCameraId(camera_id);

  const std::vector<Eigen::Vector2d> points =
      colmap::FeatureKeypointsToPointsVector(keypoints);
  image.SetPoints2D(points);

  return image;
}
