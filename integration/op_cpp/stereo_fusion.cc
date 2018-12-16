#include "io.cc"
#include <stdio.h>

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"

using colmap::Bitmap;

typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Inv_R;
typedef Eigen::Matrix<float, 3, 4, Eigen::RowMajor> Inv_P;
typedef Eigen::Matrix<float, 3, 4, Eigen::RowMajor> PMatrix;

class StereoFusionKernel : public scanner::StenciledBatchedKernel,
                           public scanner::VideoKernel {
public:
  StereoFusionKernel(const scanner::KernelConfig &config)
      : scanner::StenciledBatchedKernel(config),
        max_squared_reproj_error_(options_.max_reproj_error *
                                  options_.max_reproj_error),
        min_cos_normal_error_(std::cos(DegToRad(options_.max_normal_error))) {
    CHECK(options_.Check())
  }

  // need to get depth information first
  Mat<float> readMatFromElement(scanner::Element &el, size_t width,
                                size_t height) {
    u8 *pointer = el.buffer;
    size_t size = read_single_from_buffer<size_t>(buffer);
    buffer += sizeof(size);
    size_t depth = size / width / height;
    Mat<float> mat = Mat(width, height, depth);

    memcpy(mat.GetData().data(), reinterpret_cast<float *>(pointer),
           size * sizeof(float));

    return mat;
  }

  // converted from colmap::mvs::StereoFusion
  void execute(const scanner::BatchedElements &input_cols,
               scanner::BatchedElements &output_cols) override {
    int image_id = read_single_from_element<int>(input_cols[0]);
    size_t width = read_single_from_element<size_t>(input_cols[1]);
    size_t height = read_single_from_element<size_t>(input_cols[2]);

    Mat<float> &depth_map = readMatFromElement(input_cols[3]);
    Mat<float> &normal_map = readMatFromElement(input_cols[4]);

    float *R = readArrayFromElement<float>(input_cols[5]);
    float *K = readArrayFromElement<float>(input_cols[6]);
    float *T = readArrayFromElement<float>(input_cols[7]);

    FIBITMAP *fibitmap = FreeImage_ConvertFromRawBits(
        input_cols[8].buffer, width, height, 3 * width, 24, FI_RGBA_RED_MASK,
        FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);

    colmap::Bitmap bitmap(fibitmap);

    // following converted from StereoFusion::Run
    Mat<bool> fused_pixel_mask(width, height, 1);
    fused_pixel_mask.Fill(false);
    std::pair<int, int> depth_map_size = std::make_pair(width, height);
    std::pair<float, float> bitmap_scale = std::make_pair(1, 1);

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(K);
    K(0, 0) *= bitmap_scale.first;
    K(0, 2) *= bitmap_scale.first;
    K(1, 1) *= bitmap_scale.second;
    K(1, 2) *= bitmap_scale.second;

    PMatrix P;
    Inv_P inv_P;
    Inv_R inv_R;

    ComposeProjectionMatrix(K, R, T, P.data());
    ComposeInverseProjectionMatrix(K, R, T, inv_P.data());

    inv_R = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R)
                .transpose();

    colmap::mvs::FusionData data;
    data.image_idx = image_idx;
    data.traversal_depth = 0;
    for (data.row = 0; data.row < height; ++data.row) {
      for (data.col = 0; data.col < width; ++data.col) {
        if (fused_pixel_mask.Get(data.row, data.col)) {
          continue;
        }
        Fuse(data, fused_pixel_mask, depth_map, normal_map, P, inv_P, inv_R,
             bitmap, bitmap_scale, );
      }
    }
  }

private:
  colmap::mvs::StereoFusionOptions options_;
  const float max_squared_reproj_error_;
  const float min_cos_normal_error_;
  // converted from StereoFusion::Fuse
  void Fuse(olmap::mvs::FusionData &data, Mat<bool> &fused_pixel_mask,
            Mat<float> &depth_map, Mat<float> &normal_map, PMatrix &P, Inv_P &P,
            Inv_R &R, Bitmap &bitmap, std::pair<float, float> &bitmap_scale) {
    std::vector<FusionData> fusion_queue;

    Eigen::Vector4f fused_ref_point = Eigen::Vector4f::Zero();
    Eigen::Vector3f fused_ref_normal = Eigen::Vector3f::Zero();

    // Points of different pixels of the currently point to be fused.
    std::vector<float> fused_point_x;
    std::vector<float> fused_point_y;
    std::vector<float> fused_point_z;
    std::vector<float> fused_point_nx;
    std::vector<float> fused_point_ny;
    std::vector<float> fused_point_nz;
    std::vector<uint8_t> fused_point_r;
    std::vector<uint8_t> fused_point_g;
    std::vector<uint8_t> fused_point_b;
    std::unordered_set<int> fused_point_visibility;

    do {
      const int image_idx = data.image_idx;
      const int row = data.row;
      const int col = data.col;
      const int traversal_depth = data.traversal_depth;

      if (fused_pixel_mask.Get(row, col)) {
        continue;
      }

      const float depth = depth_map.Get(row, col);

      // Pixels with negative depth are filtered.
      if (depth <= 0.0f) {
        continue;
      }

      // If the traversal depth is greater than zero, the initial reference
      // pixel has already been added and we need to check for consistency.
      if (traversal_depth > 0) {
        // Project reference point into current view.
        const Eigen::Vector3f proj = P * fused_ref_point;

        // Depth error of reference depth with current depth.
        const float depth_error = std::abs((proj(2) - depth) / depth);
        if (depth_error > options_.max_depth_error) {
          continue;
        }

        // Reprojection error reference point in the current view.
        const float col_diff = proj(0) / proj(2) - col;
        const float row_diff = proj(1) / proj(2) - row;
        const float squared_reproj_error =
            col_diff * col_diff + row_diff * row_diff;
        if (squared_reproj_error > max_squared_reproj_error_) {
          continue;
        }
      }

      // Determine normal direction in global reference frame.
      const Eigen::Vector3f normal =
          inv_R * Eigen::Vector3f(normal_map.Get(row, col, 0),
                                  normal_map.Get(row, col, 1),
                                  normal_map.Get(row, col, 2));

      // Check for consistent normal direction with reference normal.
      if (traversal_depth > 0) {
        const float cos_normal_error = fused_ref_normal.dot(normal);
        if (cos_normal_error < min_cos_normal_error_) {
          continue;
        }
      }

      // Determine 3D location of current depth value.
      const Eigen::Vector3f xyz =
          inv_P * Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

      // Read the color of the pixel.
      colmap::BitmapColor<uint8_t> color;
      bitmap.InterpolateNearestNeighbor(col / bitmap_scale.first,
                                        row / bitmap_scale.second, &color);

      // Set the current pixel as visited.
      fused_pixel_mask.Set(row, col, true);

      // Accumulate statistics for fused point.
      fused_point_x.push_back(xyz(0));
      fused_point_y.push_back(xyz(1));
      fused_point_z.push_back(xyz(2));
      fused_point_nx.push_back(normal(0));
      fused_point_ny.push_back(normal(1));
      fused_point_nz.push_back(normal(2));
      fused_point_r.push_back(color.r);
      fused_point_g.push_back(color.g);
      fused_point_b.push_back(color.b);
      fused_point_visibility.insert(image_idx);

      // Remember the first pixel as the reference.
      if (traversal_depth == 0) {
        fused_ref_point = Eigen::Vector4f(xyz(0), xyz(1), xyz(2), 1.0f);
        fused_ref_normal = normal;
      }

      if (fused_point_x.size() >=
          static_cast<size_t>(options_.max_num_pixels)) {
        break;
      }

      FusionData next_data;
      next_data.traversal_depth = traversal_depth + 1;

      if (next_data.traversal_depth >= options_.max_traversal_depth) {
        continue;
      }

      for (const auto next_image_idx : overlapping_images_.at(image_idx)) {
        if (!used_images_.at(next_image_idx) ||
            fused_images_.at(next_image_idx)) {
          continue;
        }

        next_data.image_idx = next_image_idx;

        const Eigen::Vector3f next_proj =
            P_.at(next_image_idx) * xyz.homogeneous();
        next_data.col =
            static_cast<int>(std::round(next_proj(0) / next_proj(2)));
        next_data.row =
            static_cast<int>(std::round(next_proj(1) / next_proj(2)));

        const auto &depth_map_size = depth_map_sizes_.at(next_image_idx);
        if (next_data.col < 0 || next_data.row < 0 ||
            next_data.col >= depth_map_size.first ||
            next_data.row >= depth_map_size.second) {
          continue;
        }

        fusion_queue_.push_back(next_data);
      }
    }
  }
}

REGISTER_OP(PatchMatch)
    .stencil()
    .input("image_id")
    .input("width")
    .input("height")
    .input("R")
    .input("K")
    .input("T")
    .input("depth_map")
    .input("normal_map")
    .input("bitmap")
    .output("fused_point")
    .output("fused_point_visibility");

REGISTER_KERNEL(PatchMatch, PatchMatchKernel)
    .device(scanner::DeviceType::CPU)
    .batch()
    .num_devices(1);
