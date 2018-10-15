#include "colmap.pb.h"
#include "types.cc"

#include "scanner/api/kernel.h"
#include "scanner/util/common.h"

#include <colmap/base/database.h>
#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>

using colmap::FeatureDescriptors;
using colmap::FeatureKeypoint;
using colmap::FeatureKeypoints;
using colmap::FeatureMatch;
using colmap::FeatureMatches;
using colmap::TwoViewGeometry;

using scanner::insert_element;
using scanner::u8;

using scanner::Element;
using scanner::Elements;
using std::vector;

#define DEVICE scanner::CPU_DEVICE

struct Buffer {
  u8 *data;
  size_t size;

  Buffer(u8 *data, size_t size) : data(data), size(size) {}
};

FeatureDescriptors ReadDescriptorsFromElement(const Element &elememt) {
  u8 *descriptors_buffer = elememt.buffer;
  size_t index_size = sizeof(size_t);
  size_t rows = *(reinterpret_cast<size_t *>(descriptors_buffer));
  size_t cols = *(reinterpret_cast<size_t *>(descriptors_buffer + index_size));

  FeatureDescriptors matrix(rows, cols);
  size_t num_bytes =
      matrix.size() * sizeof(typename FeatureDescriptors::Scalar);

  memcpy(reinterpret_cast<u8 *>(matrix.data()),
         descriptors_buffer + index_size * 2, num_bytes);

  return matrix;
}

FeatureKeypoints readKeypointsFromElement(const Element &element) {
  u8 *keypoints_buffer = element.buffer;
  size_t index_size = sizeof(size_t);
  size_t rows = *(reinterpret_cast<size_t *>(keypoints_buffer));

  colmap::FeatureKeypoints keypoints(
      reinterpret_cast<FeatureKeypoint *>(keypoints_buffer + index_size),
      reinterpret_cast<FeatureKeypoint *>(keypoints_buffer + index_size +
                                          rows * sizeof(FeatureKeypoint)));

  return keypoints;
}

void writeFeatureMatchesToColumn(Elements &col, FeatureMatches &matches) {
  size_t num_matches = matches.size();
  size_t matches_byte_size = num_matches * sizeof(FeatureMatch);
  size_t total_byte_size = matches_byte_size + sizeof(num_matches);

  u8 *buffer = new_buffer(DEVICE, total_byte_size);
  memcpy(buffer, &num_matches, sizeof(num_matches));
  memcpy(buffer + sizeof(num_matches), matches.data(), matches_byte_size);

  insert_element(col, buffer, total_byte_size);
}

// read and write for vector type data
template <typename VectorType>
VectorType readVectorFromElement(const Element &element) {
  u8 *buffer = element.buffer;
  size_t index_size = sizeof(size_t);
  size_t size = *(reinterpret_cast<size_t *>(buffer));

  return VectorType(
      reinterpret_cast<typename VectorType::value_type *>(buffer + index_size),
      reinterpret_cast<typename VectorType::value_type *>(
          buffer + index_size +
          size * sizeof(typename VectorType::value_type)));
}

template <typename VectorType> Buffer createVectorBuffer(VectorType &data) {
  size_t num_elements = data.size();
  size_t element_byte_size = sizeof(typename VectorType::value_type);
  size_t data_byte_size = num_elements * element_byte_size;
  size_t buffer_size = data_byte_size + sizeof(num_elements);

  u8 *buffer = new_buffer(DEVICE, buffer_size);
  memcpy(buffer, &num_elements, sizeof(num_elements));
  memcpy(buffer + sizeof(num_elements), data.data(), data_byte_size);

  return Buffer(buffer, buffer_size);
}

template <typename VectorType>
void writeVectorToElement(Element &output, VectorType &data) {
  Buffer buffer = createVectorBuffer(data);
  insert_element(output, buffer.data, buffer.size);
}

// read and write for fixed size data
template <typename T> T readSingleFromElement(const Element &element) {
  u8 *buffer = element.buffer;
  return *(reinterpret_cast<T *>(buffer));
}

template <typename T> Buffer createSingleBuffer(T &data) {
  size_t byte_size = sizeof(T);
  u8 *buffer = new_buffer(DEVICE, byte_size);
  memcpy(buffer, &data, byte_size);
  return Buffer(buffer, byte_size);
}

template <typename T> void writeSingleToColumn(Elements &col, T &data) {
  Buffer buffer = createSingleBuffer(data);
  insert_element(col, buffer.data, buffer.size);
}

template <typename T> void writeSingleToElement(Element &element, T &data) {
  Buffer buffer = createSingleBuffer(data);
  insert_element(element, buffer.data, buffer.size);
}

// read and write for matrix type data
template <typename MatrixType>
MatrixType ReadMatrixFromElement(const Element &elememt) {
  u8 *buffer = elememt.buffer;
  size_t index_size = sizeof(size_t);
  size_t rows = *(reinterpret_cast<size_t *>(buffer));
  size_t cols = *(reinterpret_cast<size_t *>(buffer + index_size));

  MatrixType matrix(rows, cols);
  size_t matrix_num_bytes = matrix.size() * sizeof(typename MatrixType::Scalar);

  memcpy(reinterpret_cast<u8 *>(matrix.data()), buffer + index_size * 2,
         matrix_num_bytes);

  return matrix;
}

template <typename MatrixType> Buffer createMatrixBuffer(MatrixType &matrix) {
  size_t rows = matrix.rows();
  size_t cols = matrix.cols();
  size_t index_size = sizeof(size_t);

  size_t matrix_num_bytes = matrix.size() * sizeof(typename MatrixType::Scalar);

  size_t total_byte_size = matrix_num_bytes + index_size * 2;
  u8 *buffer = new_buffer(DEVICE, total_byte_size);
  std::memcpy(buffer, &rows, index_size);
  std::memcpy(buffer + index_size, &cols, index_size);
  std::memcpy(buffer + index_size * 2, matrix.data(), matrix_num_bytes);

  return Buffer(buffer, total_byte_size);
}

template <typename MatrixType>
void writeMatrixToElement(Element &element, MatrixType &matrix) {
  Buffer buffer = createMatrixBuffer(matrix);
  insert_element(element, buffer.data, buffer.size);
}

// combine a list of buffers into one buffer
Buffer combine_buffers(vector<Buffer> buffers) {
  size_t total_buffer_size = 0;
  for (int i = 0; i < buffers.size(); i++) {
    total_buffer_size += buffers[i].size;
  }

  u8 *combined = new_buffer(DEVICE, total_buffer_size);
  u8 *pointer = combined;

  for (int i = 0; i < buffers.size(); i++) {
    memcpy(pointer, buffers[i].data, buffers[i].size);
    pointer += buffers[i].size;
    delete_buffer(DEVICE, buffers[i].data);
  }

  return Buffer(combined, total_buffer_size);
}

// write sequential matching result to scanner element
// TODO: create only one buffer
void writeMatchingResultToColumn(Elements &col, MatchingResult &result) {
  vector<Buffer> buffers;
  buffers.push_back(createSingleBuffer(result.image));

  int num_peers = result.peers.size();
  buffers.push_back(createSingleBuffer(num_peers));

  // create buffers for matches, each matches is a vector
  for (int i = 0; i < num_peers; i++) {
    buffers.push_back(createVectorBuffer(result.matches_list[i]));
  }

  // create buffers for two view geometries
  for (int i = 0; i < num_peers; i++) {
    buffers.push_back(createSingleBuffer(result.two_view_geometry_list[i]));
  }

  Buffer combined = combine_buffers(buffers);
  insert_element(col, combined.data, combined.size);
}
