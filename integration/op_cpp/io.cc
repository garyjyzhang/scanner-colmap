#include "scanner/api/kernel.h"
#include "scanner/util/common.h"

#include <colmap/base/database.h>
#include <colmap/base/point2d.h>
#include <colmap/base/reconstruction.h>
#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>
#include <colmap/util/misc.h>
#include <stdlib.h>

using colmap::Camera;
using colmap::FeatureDescriptors;
using colmap::FeatureKeypoint;
using colmap::FeatureKeypoints;
using colmap::FeatureMatch;
using colmap::FeatureMatches;
using colmap::Reconstruction;
using colmap::TwoViewGeometry;

using scanner::Element;
using scanner::Elements;
using scanner::insert_element;
using scanner::u8;
using scanner::DeviceHandle;

using std::vector;

#define DEVICE scanner::CPU_DEVICE

struct Buffer {
  u8 *data;
  size_t size;

  Buffer(u8 *data, size_t size) : data(data), size(size) {}
};

// copy the content of the object to the buffer and return the
// incremented buffer pointer
template <typename T> u8 *copy_object_to_buffer(void *buffer, T &data) {
  memcpy(buffer, &data, sizeof(T));
  return static_cast<u8 *>(buffer) + sizeof(T);
}

// copy content from source buffer to destination buffer
// return the incremented buffer pointer
u8 *copy_buffer_to_buffer(void *dest, const void *src, size_t size) {
  memcpy(dest, src, size);
  return static_cast<u8 *>(dest) + size;
}

// read a single value of type T from the buffer
// return the value
template <typename T> T read_single_from_buffer(u8 *buffer) {
  return *(reinterpret_cast<T *>(buffer));
}

// read a single value of type T from the buffer
// return the incremented buffer
template <typename T> u8 *read_single_from_buffer(u8 *buffer, T *obj_ptr) {
  memcpy(static_cast<void *>(obj_ptr), static_cast<void *>(buffer), sizeof(T));
  return buffer + sizeof(T);
}

// read a single value of type T from a scanner Element object
// return the value
template <typename T> T read_single_from_element(const Element &element) {
  return read_single_from_buffer<T>(element.buffer);
}

// create a Buffer for a single element using the scanner::new_buffer
template <typename T>
Buffer create_single_buffer(T &data, const DeviceHandle &device = DEVICE) {
  size_t byte_size = sizeof(T);
  u8 *buffer = new_buffer(device, byte_size);
  memcpy_buffer(buffer, device, reinterpret_cast<const u8 *>(&data), device,
                byte_size);
  return Buffer(buffer, byte_size);
}

// write a single value to a scanner column
template <typename T>
void write_single_to_column(Elements &col, const T &data,
                            const DeviceHandle &device = DEVICE) {
  Buffer buffer = create_single_buffer(data, device);
  insert_element(col, buffer.data, buffer.size);
}

// write a single value to a scanner element
template <typename T>
void write_single_to_element(Element &element, const T &data,
                             const DeviceHandle &device = DEVICE) {
  Buffer buffer = create_single_buffer(data, device);
  insert_element(element, buffer.data, buffer.size);
}

// write a single element to column and insert extra refs - 1 empty elements
// this is used for kernels that performs a reduction
// a block buffer is allocated to be able to perform sampling on the output to
// remove the empty elements
template <typename T>
void write_single_and_fill_column(Elements &col, T &data, int refs) {
  size_t byte_size = sizeof(T);
  u8 *buffer = new_block_buffer(DEVICE, byte_size, refs);
  memcpy(buffer, &data, byte_size);
  insert_element(col, buffer, byte_size);
  for (int i = 1; i < refs; i++) {
    insert_element(col, buffer, 0);
  }
}

// read a vector from the buffer, it is assumed that the size of the vector
// is stored as size_t at the start of the buffer
// return the vector
template <typename VectorType> VectorType read_vector_from_buffer(u8 *buffer) {
  size_t size = read_single_from_buffer<size_t>(buffer);
  buffer += sizeof(size);

  return VectorType(
      reinterpret_cast<typename VectorType::value_type *>(buffer),
      reinterpret_cast<typename VectorType::value_type *>(
          buffer + size * sizeof(typename VectorType::value_type)));
}

// read a vector from the buffer, it is assumed that the size of the vector
// is stored as size_t at the start of the buffer
// return the incremented buffer pointer
template <typename VectorType>
u8 *read_vector_from_buffer(u8 *buffer, VectorType *vector_ptr) {
  size_t size = read_single_from_buffer<size_t>(buffer);
  buffer += sizeof(size);
  size_t data_byte_size = size * sizeof(typename VectorType::value_type);

  *vector_ptr =
      VectorType(reinterpret_cast<typename VectorType::value_type *>(buffer),
                 reinterpret_cast<typename VectorType::value_type *>(
                     buffer + data_byte_size));

  return buffer + data_byte_size;
}

// read a vector type from a scanner element
// return the read vector
template <typename VectorType>
VectorType read_vector_from_element(const Element &element) {
  return read_vector_from_buffer<VectorType>(element.buffer);
}

// create a Buffer for a vector type, a size_t is inserted at the start
// of the buffer to indicate the size of vector
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

// write a vector type to a scanner element
template <typename VectorType>
void write_vector_to_element(Element &output, VectorType &data) {
  Buffer buffer = createVectorBuffer(data);
  insert_element(output, buffer.data, buffer.size);
}

// write a vector type to a scanner column
template <typename VectorType>
void write_vector_to_column(Elements &col, const VectorType &data) {
  Buffer buffer = createVectorBuffer(data);
  insert_element(col, buffer.data, buffer.size);
}

// read a matrix type from a scanner element, it is assumed that the number of
// rows and columns of the matrix is stored at the start of the buffer
template <typename MatrixType>
MatrixType read_matrix_from_element(const Element &elememt) {
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

// create a Buffer for a matrix type. The number of rows and columns will be
// stored at the start of the buffer
template <typename MatrixType> Buffer create_matrix_buffer(MatrixType &matrix) {
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

// write a matrix type to a scanner element
template <typename MatrixType>
void write_matrix_to_element(Element &element, MatrixType &matrix) {
  Buffer buffer = create_matrix_buffer(matrix);
  insert_element(element, buffer.data, buffer.size);
}

// read a list of two view geometry from a scanner element
// the first position of the buffer should be the total byte size of for sanity
// check the second position should be the number of tvgs
vector<TwoViewGeometry> read_two_view_geometries(const Element &element) {
  u8 *buffer = element.buffer;

  size_t total_byte_size;
  buffer = read_single_from_buffer<size_t>(buffer, &total_byte_size);

  int num_geometries;
  buffer = read_single_from_buffer<int>(buffer, &num_geometries);

  vector<TwoViewGeometry> tvgs(num_geometries);

  for (int i = 0; i < num_geometries; i++) {
    TwoViewGeometry &tvg = tvgs[i];
    buffer = read_single_from_buffer<int>(buffer, &(tvg.config));
    buffer = read_single_from_buffer<Eigen::Matrix3d>(buffer, &(tvg.E));
    buffer = read_single_from_buffer<Eigen::Matrix3d>(buffer, &(tvg.F));
    buffer = read_single_from_buffer<Eigen::Matrix3d>(buffer, &(tvg.H));
    buffer = read_single_from_buffer<Eigen::Vector4d>(buffer, &(tvg.qvec));
    buffer = read_single_from_buffer<Eigen::Vector3d>(buffer, &(tvg.tvec));
    buffer = read_single_from_buffer<double>(buffer, &(tvg.tri_angle));

    buffer =
        read_vector_from_buffer<FeatureMatches>(buffer, &(tvg.inlier_matches));
  }

  assert(buffer == element.buffer + total_byte_size);
  return tvgs;
}

// create a Buffer for a list of two view geometries
// the first position of the buffer will store the total byte size for sanity
// check the second position stores the number of two view geometries
Buffer create_two_view_geometries_buffer(
    vector<TwoViewGeometry> &two_view_geometry_list) {
  int num_geometries = two_view_geometry_list.size();
  size_t total_byte_size = sizeof(num_geometries);

  for (TwoViewGeometry &tvg : two_view_geometry_list) {
    total_byte_size += sizeof(tvg.config) + sizeof(tvg.E) + sizeof(tvg.F) +
                       sizeof(tvg.H) + sizeof(tvg.qvec) + sizeof(tvg.tvec) +
                       sizeof(tvg.tri_angle);

    size_t num_matches = tvg.inlier_matches.size();
    total_byte_size += sizeof(num_matches) + num_matches * sizeof(FeatureMatch);
  }

  total_byte_size += sizeof(total_byte_size);

  u8 *buffer = new_buffer(DEVICE, total_byte_size), *pointer = buffer;

  // write total byte size for read validation
  pointer = copy_object_to_buffer(pointer, total_byte_size);
  pointer = copy_object_to_buffer(pointer, num_geometries);

  for (TwoViewGeometry &tvg : two_view_geometry_list) {
    pointer = copy_object_to_buffer(pointer, tvg.config);
    pointer = copy_object_to_buffer(pointer, tvg.E);
    pointer = copy_object_to_buffer(pointer, tvg.F);
    pointer = copy_object_to_buffer(pointer, tvg.H);
    pointer = copy_object_to_buffer(pointer, tvg.qvec);
    pointer = copy_object_to_buffer(pointer, tvg.tvec);
    pointer = copy_object_to_buffer(pointer, tvg.tri_angle);

    FeatureMatches &matches = tvg.inlier_matches;
    size_t num_matches = matches.size();
    size_t data_byte_size = num_matches * sizeof(FeatureMatch);

    pointer = copy_object_to_buffer(pointer, num_matches);
    pointer = copy_buffer_to_buffer(pointer, matches.data(), data_byte_size);
  }

  assert(pointer == buffer + total_byte_size);
  return Buffer(buffer, total_byte_size);
}

// write a list of two view geometries to a scanner column
void write_two_view_geometries_to_column(Elements &col,
                                         vector<TwoViewGeometry> &tvg_list) {
  Buffer buffer = create_two_view_geometries_buffer(tvg_list);
  insert_element(col, buffer.data, buffer.size);
}

// create a Buffer for a colmap scanner object
Buffer create_camera_buffer(Camera &camera) {
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

// write a colmap camera object to scanner element
void write_camera_to_element(Element &element, Camera &camera) {
  Buffer buffer = create_camera_buffer(camera);
  insert_element(element, buffer.data, buffer.size);
}

// read a colmap camera object from a scanner element
void read_camera_from_element(const Element &element, Camera *camera) {
  u8 *buffer = element.buffer;
  size_t total_byte_size;
  colmap::camera_t camera_id;
  int model_id;
  size_t width;
  size_t height;
  vector<double> params;
  bool prior_focal_length;

  buffer = read_single_from_buffer<size_t>(buffer, &total_byte_size);
  buffer = read_single_from_buffer<colmap::camera_t>(buffer, &camera_id);
  buffer = read_single_from_buffer<int>(buffer, &model_id);
  buffer = read_single_from_buffer<size_t>(buffer, &width);
  buffer = read_single_from_buffer<size_t>(buffer, &height);
  buffer = read_single_from_buffer<bool>(buffer, &prior_focal_length);

  buffer = read_vector_from_buffer<vector<double>>(buffer, &(camera->Params()));

  assert(buffer == element.buffer + total_byte_size);

  camera->SetCameraId(camera_id);
  camera->SetModelId(model_id);
  camera->SetWidth(width);
  camera->SetHeight(height);
  camera->SetPriorFocalLength(prior_focal_length);
}

// read a list of colmap keypoints from scanner element
// it is assumed that the first position of the buffer stores the number of
// keypoints as size_t
FeatureKeypoints read_keypoints_from_element(const Element &element) {
  u8 *buffer = element.buffer;
  size_t num_keypoints;
  buffer = read_single_from_buffer<size_t>(buffer, &num_keypoints);

  colmap::FeatureKeypoints keypoints(
      reinterpret_cast<FeatureKeypoint *>(buffer),
      reinterpret_cast<FeatureKeypoint *>(
          buffer + num_keypoints * sizeof(FeatureKeypoint)));

  return keypoints;
}

// write a binary file to scanner column
// this function will also fill the column with refs number of empty elements
void write_binary_file_to_column(Elements &column, std::string path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open());

  file.seekg(0, file.end);
  size_t data_byte_size = file.tellg();
  file.seekg(0, file.beg);

  size_t total_byte_size = data_byte_size + sizeof(data_byte_size);

  // use block buffer here because we want to insert empty elements into the
  // column, which will be removed in the subsequent sampling stepts
  u8 *buffer = new_buffer(DEVICE, total_byte_size), *pointer = buffer;
  pointer = copy_object_to_buffer(buffer, data_byte_size);
  CHECK(file.read(reinterpret_cast<char *>(pointer), data_byte_size));
  file.close();

  insert_element(column, buffer, total_byte_size);
}

// write a reconstruction to 3 separate columns for cameras, images and points3d
// note that the columns will be filled with refs empty columns at the end
void write_reconstruction_to_columns(Elements &cameras, Elements &images,
                                     Elements &points3d, std::string path) {
  write_binary_file_to_column(cameras, path + "/cameras.bin");
  write_binary_file_to_column(images, path + "/images.bin");
  write_binary_file_to_column(points3d, path + "/points3D.bin");
}

// write the binary content within a scanner element to a file
// this is used to read reconstruction from scanner using colmap's
// builtin reconstruction reader. Hacky but the easiest way to achieve that.
void write_binary_to_file(const Element &element, std::string path) {
  u8 *buffer = element.buffer;
  size_t size;
  buffer = read_single_from_buffer<size_t>(buffer, &size);

  std::ofstream file(path, std::ios::binary);
  CHECK(file.write(reinterpret_cast<char *>(buffer), size));
  file.close();
}

// write the content of elements into binary files and use colmap's
// builtin reconstruction reader to recreate the reconstruction
// this is a litte messy but is the easiest way to get the job done
void loadReconstruction(const Element &cameras, const Element &images,
                        const Element &points3d,
                        Reconstruction &reconstruction) {
  // use the address of reconstruction object as tmp folder name
  size_t addr = reinterpret_cast<size_t>(&reconstruction);

  std::string tmp_path = std::to_string(addr);
  colmap::CreateDirIfNotExists(tmp_path);

  // convert binary content of each element to a file
  write_binary_to_file(cameras, tmp_path + "/cameras.bin");
  write_binary_to_file(images, tmp_path + "/images.bin");
  write_binary_to_file(points3d, tmp_path + "/points3D.bin");

  // create the reconstruction
  reconstruction.ReadBinary(tmp_path);

  // remove the tmp files
  CHECK_EQ(system(("rm -rf " + tmp_path).c_str()), 0);
}

void writeImageToColumn(Elements &col, colmap::Image &image,
                        scanner::DeviceHandle device = DEVICE) {
  const auto &image_id = image.ImageId();
  const auto &camera_id = image.CameraId();
  const auto &qvec = image.Qvec();
  const auto &tvec = image.Tvec();
  const auto &num_points_2d = image.NumPoints2D();
  const auto &points2d = image.Points2D();

  size_t buffer_size = sizeof(image_id) + sizeof(camera_id) + sizeof(qvec) +
                       sizeof(tvec) + sizeof(num_points_2d) +
                       sizeof(colmap::Point2D) * num_points_2d;

  u8 *buffer = new_buffer(device, buffer_size), *pointer = buffer;
  pointer = copy_object_to_buffer(pointer, image_id);
  pointer = copy_object_to_buffer(pointer, camera_id);
  pointer = copy_object_to_buffer(pointer, qvec);
  pointer = copy_object_to_buffer(pointer, tvec);
  pointer = copy_object_to_buffer(pointer, num_points_2d);
  for (const auto &point2d : points2d) {
    pointer = copy_object_to_buffer(pointer, point2d);
  }

  insert_element(col, buffer, buffer_size);
}

void writeCameraToColumn(Elements &col, colmap::Camera &camera,
                         scanner::DeviceHandle device = DEVICE) {
  const auto &camera_id = camera.CameraId();
  const auto &model_id = camera.ModelId();
  const auto &width = camera.Width();
  const auto &height = camera.Height();
  const vector<double> &params = camera.Params();
  const auto &prior_focal_length = camera.HasPriorFocalLength();

  size_t buffer_size = sizeof(camera_id) + sizeof(model_id) + sizeof(width) +
                       sizeof(height) + params.size() * sizeof(double);

  u8 *buffer = new_buffer(device, buffer_size), *pointer = buffer;
  pointer = copy_object_to_buffer(pointer, camera_id);
  pointer = copy_object_to_buffer(pointer, model_id);
  pointer = copy_object_to_buffer(pointer, width);
  pointer = copy_object_to_buffer(pointer, height);
  for (const double &param : params) {
    pointer = copy_object_to_buffer(pointer, param);
  }

  insert_element(col, buffer, buffer_size);
}

colmap::Image readImageFromColumn(const Element data) {
  u8 *buffer = data.buffer, *pointer = buffer;
  colmap::Image image;

  colmap::image_t image_id;
  colmap::camera_t camera_id;
  Eigen::Vector4d qvec;
  Eigen::Vector3d tvec;
  colmap::point2D_t num_points2d;
  vector<colmap::Point2D> points2d;

  pointer = read_single_from_buffer(pointer, &image_id);
  pointer = read_single_from_buffer(pointer, &camera_id);
  pointer = read_single_from_buffer(pointer, &qvec);
  pointer = read_single_from_buffer(pointer, &tvec);
  pointer = read_single_from_buffer(pointer, &num_points2d);

  std::vector<colmap::point3D_t> point3D_ids;
  point3D_ids.reserve(num_points2d);
  for (int i = 0; i < num_points2d; i++) {
    colmap::Point2D point2d;
    pointer = read_single_from_buffer(pointer, &point2d);
    points2d.push_back(point2d);
  }

  image.SetImageId(image_id);
  image.SetCameraId(camera_id);
  image.SetQvec(qvec);
  image.SetTvec(tvec);
  image.SetPoints2D(points2d);

  image.NormalizeQvec();
  // image.SetUp(Camera(image.CameraId()));
  image.SetRegistered(true);

  return image;
}

colmap::Camera readCameraFromColumn(const Element data) {
  u8 *buffer = data.buffer, *pointer = buffer;
  colmap::Camera camera;

  colmap::camera_t camera_id;
  int model_id;
  size_t width;
  size_t height;
  bool prior_focal_length;

  pointer = read_single_from_buffer(pointer, &camera_id);
  pointer = read_single_from_buffer(pointer, &model_id);
  pointer = read_single_from_buffer(pointer, &width);
  pointer = read_single_from_buffer(pointer, &height);

  camera.SetCameraId(camera_id);
  camera.SetModelId(model_id);
  camera.SetWidth(width);
  camera.SetHeight(height);

  auto params = camera.Params();
  for (int i = 0; i < params.size(); i++) {
    double param;
    pointer = read_single_from_buffer(pointer, &param);
    params[i] = param;
  }

  assert(camera.VerifyParams());
  return camera;
}

template <typename ElementType>
void writeArrayToColumn(Elements &col, const ElementType *array, int size,
                        const scanner::DeviceHandle &device) {
  size_t buffer_size = size * sizeof(ElementType);

  u8 *buffer = new_buffer(device, buffer_size), *pointer = buffer;
  memcpy(pointer, reinterpret_cast<const u8 *>(array), buffer_size);

  insert_element(col, buffer, buffer_size);
}

template <typename Type> Type *readArrayFromElement(const Element &el) {
  return reinterpret_cast<Type *>(el.buffer);
}
