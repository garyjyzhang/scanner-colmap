#include "colmap.pb.h"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"
#include "scanner/util/memory.h"

#include <colmap/base/database.h>
#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>

using colmap::FeatureDescriptors;
using colmap::FeatureKeypoint;
using colmap::FeatureKeypoints;
using colmap::FeatureMatches;
using colmap::image_t;
using colmap::TwoViewGeometry;

using scanner::insert_element;
using scanner::u8;

FeatureDescriptors ReadDescriptorsFromElement(const scanner::Element &elememt) {
  u8 *descriptors_buffer = elememt.buffer;
  size_t index_size = sizeof(size_t);
  size_t rows = *(reinterpret_cast<size_t *>(descriptors_buffer));
  size_t cols = *(reinterpret_cast<size_t *>(descriptors_buffer + index_size));
  std::cout << "rows, cols: " << rows << " " << cols << std::endl;
  FeatureDescriptors matrix(rows, cols);
  size_t num_bytes =
      matrix.size() * sizeof(typename FeatureDescriptors::Scalar);

  std::memcpy(reinterpret_cast<u8 *>(matrix.data()),
              descriptors_buffer + index_size * 2, num_bytes);

  return matrix;
}

FeatureKeypoints readKeypointsFromElement(const scanner::Element &element) {
  u8 *keypoints_buffer = element.buffer;
  size_t index_size = sizeof(size_t);
  size_t rows = *(reinterpret_cast<size_t *>(keypoints_buffer));

  colmap::FeatureKeypoints keypoints(
      reinterpret_cast<FeatureKeypoint *>(keypoints_buffer + index_size),
      reinterpret_cast<FeatureKeypoint *>(keypoints_buffer + index_size +
                                          rows * sizeof(FeatureKeypoint)));

  return keypoints;
}

void writeFeatureMatchesToColumn(scanner::Elements &col,
                                 FeatureMatches &matches) {
  size_t num_matches = matches.size();
  size_t matches_byte_size = num_matches * sizeof(colmap::FeatureMatch);
  size_t total_byte_size = matches_byte_size + sizeof(num_matches);

  u8 *buffer = new_buffer(scanner::CPU_DEVICE, total_byte_size);
  std::memcpy(buffer, &num_matches, sizeof(num_matches));
  std::memcpy(buffer + sizeof(num_matches), matches.data(), matches_byte_size);

  insert_element(col, buffer, total_byte_size);
}

template <typename T> void writeSingleToColumn(scanner::Elements &col, T data) {
  size_t byte_size = sizeof(T);
  u8 *buffer = new_buffer(scanner::CPU_DEVICE, byte_size);
  std::memcpy(buffer, &data, byte_size);
  insert_element(col, buffer, byte_size);
}
