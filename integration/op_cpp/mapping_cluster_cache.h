#ifndef CLUSTER_MAPPING_CACHE_H_
#define CLUSTER_MAPPING_CACHE_H_
#include "io.cc"

#include <colmap/base/database_cache.h>

using colmap::Camera;
using colmap::camera_t;
using colmap::CorrespondenceGraph;
using colmap::Image;
using colmap::image_t;
using scanner::Elements;

typedef vector<image_t> PairImageIds;
typedef vector<TwoViewGeometry> TwoViewGeometries;

class MappingClusterCache : public colmap::DatabaseCache {
public:
  void LoadFromStencils(const Elements &image_id_stencil,
                        const Elements &pair_image_id_stencil,
                        const Elements &two_view_geometry_stencil,
                        const Elements &keypoints_stencil,
                        const Elements &camera_stencil, int num_pivot_images,
                        const size_t min_num_matches,
                        const bool ignore_watermarks);

  // Get number of objects.
  inline size_t NumCameras() const;
  inline size_t NumImages() const;

  // Get specific objects.
  inline class Camera &Camera(const camera_t camera_id);
  inline const class Camera &Camera(const camera_t camera_id) const;
  inline class Image &Image(const image_t image_id);
  inline const class Image &Image(const image_t image_id) const;

  // Get all objects.
  inline const EIGEN_STL_UMAP(camera_t, class Camera) & Cameras() const;
  inline const EIGEN_STL_UMAP(image_t, class Image) & Images() const;

  // Check whether specific object exists.
  inline bool ExistsCamera(const camera_t camera_id) const;
  inline bool ExistsImage(const image_t image_id) const;

  // Get reference to correspondence graph.
  inline const class CorrespondenceGraph &CorrespondenceGraph() const;

  // Manually add data to cache.
  void AddCamera(const class Camera &camera);
  void AddImage(const class Image &image);

private:
  colmap::Image createImage(image_t id, colmap::camera_t camera_id,
                            FeatureKeypoints &keypoints);

  class CorrespondenceGraph correspondence_graph;
  EIGEN_STL_UMAP(camera_t, class Camera) cameras;
  EIGEN_STL_UMAP(image_t, class Image) images;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t MappingClusterCache::NumCameras() const { return cameras.size(); }
size_t MappingClusterCache::NumImages() const { return images.size(); }

class Camera &MappingClusterCache::Camera(const camera_t camera_id) {
  return cameras.at(camera_id);
}

const class Camera &
MappingClusterCache::Camera(const camera_t camera_id) const {
  return cameras.at(camera_id);
}

class Image &MappingClusterCache::Image(const image_t image_id) {
  return images.at(image_id);
}

const class Image &MappingClusterCache::Image(const image_t image_id) const {
  return images.at(image_id);
}

const EIGEN_STL_UMAP(camera_t, class Camera) &
    MappingClusterCache::Cameras() const {
  return cameras;
}

const EIGEN_STL_UMAP(image_t, class Image) &
    MappingClusterCache::Images() const {
  return images;
}

bool MappingClusterCache::ExistsCamera(const camera_t camera_id) const {
  return cameras.find(camera_id) != cameras.end();
}

bool MappingClusterCache::ExistsImage(const image_t image_id) const {
  return images.find(image_id) != images.end();
}

inline const class CorrespondenceGraph &
MappingClusterCache::CorrespondenceGraph() const {
  return correspondence_graph;
}

#endif
