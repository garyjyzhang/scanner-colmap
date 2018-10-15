#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"
#include "scanner/util/memory.h"

#include <colmap/base/image_reader.h>
#include <colmap/feature/sift.h>
#include <colmap/feature/types.h>
#include <colmap/util/bitmap.h>
#include <colmap/util/option_manager.h>

using colmap::FeatureMatches;
using colmap::Image;
using colmap::image_t;
using colmap::TwoViewGeometry;

struct MatchingResult {
  Image image;
  std::vector<Image> peers;
  std::vector<FeatureMatches> matches_list;
  std::vector<TwoViewGeometry> two_view_geometry_list;

  MatchingResult(Image image) : image(image) {}
  void add_peer(Image &peer) { peers.push_back(peer); }

  void add_feature_matches(FeatureMatches &matches) {
    matches_list.push_back(matches);
  }

  void add_two_view_geometry(TwoViewGeometry &geometry) {
    two_view_geometry_list.push_back(geometry);
  }
};
