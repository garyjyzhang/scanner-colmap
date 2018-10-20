#include "scanner/api/kernel.h"
#include "scanner/util/common.h"

#include <colmap/base/database.h>
#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>

using colmap::Camera;
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

// copy data to buffer and return incremented pointer
template <typename T> u8 *copy_object_to_buffer(void *buffer, T &data) {
  memcpy(buffer, &data, sizeof(T));
  return static_cast<u8 *>(buffer) + sizeof(T);
}

u8 *copy_buffer_to_buffer(void *dest, void *src, size_t size) {
  memcpy(dest, src, size);
  return static_cast<u8 *>(dest) + size;
}

// read and write for fixed size data
template <typename T> T readSingleFromBuffer(u8 *buffer) {
  return *(reinterpret_cast<T *>(buffer));
}

template <typename T> u8 *readSingleFromBuffer(u8 *buffer, T *obj_ptr) {
  memcpy(static_cast<void *>(obj_ptr), static_cast<void *>(buffer), sizeof(T));
  return buffer + sizeof(T);
}

template <typename T> T readSingleFromElement(const Element &element) {
  return readSingleFromBuffer<T>(element.buffer);
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
template <typename VectorType> VectorType readVectorFromBuffer(u8 *buffer) {
  size_t size = readSingleFromBuffer<size_t>(buffer);
  buffer += sizeof(size);

  return VectorType(
      reinterpret_cast<typename VectorType::value_type *>(buffer),
      reinterpret_cast<typename VectorType::value_type *>(
          buffer + size * sizeof(typename VectorType::value_type)));
}

template <typename VectorType>
u8 *readVectorFromBuffer(u8 *buffer, VectorType *vector_ptr) {
  size_t size = readSingleFromBuffer<size_t>(buffer);
  buffer += sizeof(size);
  size_t data_byte_size = size * sizeof(typename VectorType::value_type);

  *vector_ptr =
      VectorType(reinterpret_cast<typename VectorType::value_type *>(buffer),
                 reinterpret_cast<typename VectorType::value_type *>(
                     buffer + data_byte_size));

  return buffer + data_byte_size;
}

template <typename VectorType>
VectorType readVectorFromElement(const Element &element) {
  return readVectorFromBuffer<VectorType>(element.buffer);
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

template <typename VectorType>
void writeVectorToColumn(Elements &col, VectorType &data) {
  Buffer buffer = createVectorBuffer(data);
  insert_element(col, buffer.data, buffer.size);
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

// create buffer for a list of feature matches in two passes
Buffer createFeatureMatchesList(vector<FeatureMatches> &matches_list) {
  size_t size = matches_list.size();
  size_t total_byte_size = sizeof(size);
  size_t index_size = sizeof(size_t);
  size_t feature_match_size = sizeof(FeatureMatch);

  for (FeatureMatches &matches : matches_list) {
    size_t num_matches = matches.size();
    total_byte_size += index_size + num_matches * feature_match_size;
  }

  u8 *buffer = new_buffer(DEVICE, total_byte_size);
  u8 *pointer = buffer;

  memcpy(pointer, &size, index_size);
  pointer += index_size;

  for (FeatureMatches &matches : matches_list) {
    size_t num_matches = matches.size();
    size_t data_byte_size = num_matches * feature_match_size;

    memcpy(pointer, &data_byte_size, index_size);
    pointer += index_size;
    memcpy(pointer, matches.data(), data_byte_size);
    pointer += data_byte_size;
  }

  assert(pointer == buffer + total_byte_size);
  return Buffer(buffer, total_byte_size);
}

void writeFeatureMatchesListToColumn(Elements &col,
                                     vector<FeatureMatches> &matches_list) {
  Buffer buffer = createFeatureMatchesList(matches_list);
  insert_element(col, buffer.data, buffer.size);
}

// read and write for two view geometries
vector<TwoViewGeometry> readTwoViewGeometries(const Element &element) {
  u8 *buffer = element.buffer;

  size_t total_byte_size = readSingleFromBuffer<size_t>(buffer);
  buffer += sizeof(total_byte_size);
  size_t num_geometris = readSingleFromBuffer<size_t>(buffer);
  buffer += sizeof(num_geometris);
  size_t object_byte_size = num_geometris * sizeof(TwoViewGeometry);

  // read tvg objects
  vector<TwoViewGeometry> tvgs(
      reinterpret_cast<TwoViewGeometry *>(buffer),
      reinterpret_cast<TwoViewGeometry *>(buffer + object_byte_size));
  buffer += object_byte_size;

  // read feature matches
  for (int i = 0; i < num_geometris; i++) {
    size_t num_matches = readSingleFromBuffer<size_t>(buffer);
    buffer += sizeof(num_matches);

    size_t matches_byte_size = num_matches * sizeof(FeatureMatch);
    tvgs[i].inlier_matches = vector<FeatureMatch>(
        reinterpret_cast<FeatureMatch *>(buffer),
        reinterpret_cast<FeatureMatch *>(buffer + matches_byte_size));
    buffer += matches_byte_size;
  }

  assert(buffer == element.buffer + total_byte_size);
  return tvgs;
}

// Create a buffer for a list of two view geometries
// need to store the object and the feature matches separately
Buffer createTwoViewGeometryListBuffer(
    vector<TwoViewGeometry> &two_view_geometry_list) {
  size_t index_byte_size = sizeof(size_t);
  size_t num_geometries = two_view_geometry_list.size();
  size_t total_byte_size = 0;
  size_t object_byte_size = sizeof(TwoViewGeometry) * num_geometries;

  // total byte size + num_geometris + geometry objects
  total_byte_size += index_byte_size * 2 + object_byte_size;

  for (int i = 0; i < num_geometries; i++) {
    FeatureMatches &matches = two_view_geometry_list[i].inlier_matches;
    // feature matches data
    total_byte_size += index_byte_size + matches.size() * sizeof(FeatureMatch);
  }

  u8 *buffer = new_buffer(DEVICE, total_byte_size), *pointer = buffer;

  // write total byte size for read validation
  pointer = copy_object_to_buffer(pointer, total_byte_size);

  pointer = copy_object_to_buffer(pointer, num_geometries);

  pointer = copy_buffer_to_buffer(pointer, two_view_geometry_list.data(),
                                  object_byte_size);

  for (int i = 0; i < num_geometries; i++) {
    FeatureMatches &matches = two_view_geometry_list[i].inlier_matches;
    size_t num_matches = matches.size();
    size_t data_byte_size = matches.size() * sizeof(FeatureMatch);

    pointer = copy_object_to_buffer(pointer, num_matches);
    pointer = copy_buffer_to_buffer(pointer, matches.data(), data_byte_size);
  }

  assert(pointer == buffer + total_byte_size);
  return Buffer(buffer, total_byte_size);
}

void writeTwoViewGeometryListToColumn(Elements &col,
                                      vector<TwoViewGeometry> &tvg_list) {
  Buffer buffer = createTwoViewGeometryListBuffer(tvg_list);
  insert_element(col, buffer.data, buffer.size);
}

Buffer createCameraBuffer(Camera &camera) {
  colmap::camera_t camera_id = camera.CameraId();
  int model_id = camera.ModelId();
  size_t width = camera.Width();
  size_t height = camera.Height();
  vector<double> &params = camera.Params();
  bool prior_focal_length = camera.HasPriorFocalLength();
  size_t num_params = params.size();

  size_t params_byte_size = num_params * sizeof(double);

  size_t total_byte_size =
      sizeof(camera_id) + sizeof(model_id) + sizeof(width) + sizeof(height) +
      sizeof(prior_focal_length) + sizeof(num_params) + params_byte_size;
  total_byte_size += sizeof(total_byte_size);

  u8 *buffer = new_buffer(DEVICE, total_byte_size), *pointer = buffer;
  pointer = copy_object_to_buffer(pointer, total_byte_size);
  pointer = copy_object_to_buffer(pointer, camera_id);
  pointer = copy_object_to_buffer(pointer, model_id);
  pointer = copy_object_to_buffer(pointer, width);
  pointer = copy_object_to_buffer(pointer, height);
  pointer = copy_object_to_buffer(pointer, prior_focal_length);
  pointer = copy_object_to_buffer(pointer, num_params);
  pointer = copy_buffer_to_buffer(pointer, params.data(), params_byte_size);

  assert(pointer == buffer + total_byte_size);
  return Buffer(buffer, total_byte_size);
}

void writeCameraToElement(Element &element, Camera &camera) {
  Buffer buffer = createCameraBuffer(camera);
  insert_element(element, buffer.data, buffer.size);
}

void readCameraFromElement(const Element &element, Camera *camera) {
  u8 *buffer = element.buffer;
  size_t total_byte_size;
  colmap::camera_t camera_id;
  int model_id;
  size_t width;
  size_t height;
  vector<double> params;
  bool prior_focal_length;

  buffer = readSingleFromBuffer<size_t>(buffer, &total_byte_size);
  buffer = readSingleFromBuffer<colmap::camera_t>(buffer, &camera_id);
  buffer = readSingleFromBuffer<int>(buffer, &model_id);
  buffer = readSingleFromBuffer<size_t>(buffer, &width);
  buffer = readSingleFromBuffer<size_t>(buffer, &height);
  buffer = readSingleFromBuffer<bool>(buffer, &prior_focal_length);

  buffer = readVectorFromBuffer<vector<double>>(buffer, &(camera->Params()));

  assert(buffer == element.buffer + total_byte_size);

  camera->SetCameraId(camera_id);
  camera->SetModelId(model_id);
  camera->SetWidth(width);
  camera->SetHeight(height);
  camera->SetPriorFocalLength(prior_focal_length);
}

FeatureKeypoints readKeypointsFromElement(const Element &element) {
  u8 *buffer = element.buffer;
  size_t num_keypoints;
  buffer = readSingleFromBuffer<size_t>(buffer, &num_keypoints);

  colmap::FeatureKeypoints keypoints(
      reinterpret_cast<FeatureKeypoint *>(buffer),
      reinterpret_cast<FeatureKeypoint *>(
          buffer + num_keypoints * sizeof(FeatureKeypoint)));

  return keypoints;
}
