#include "io.cc"
#include <stdio.h>

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"

#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>
#include <colmap/mvs/depth_map.h>
#include <colmap/mvs/image.h>
#include <colmap/mvs/normal_map.h>
#include <colmap/mvs/patch_match.h>
#include <colmap/util/cuda.h>

using colmap::image_t;
using colmap::Bitmap;
using colmap::mvs::Image;
using colmap::mvs::Mat;
using colmap::mvs::DepthMap;
using colmap::mvs::NormalMap;
using colmap::mvs::PatchMatch;

class PatchMatchKernel : public scanner::StenciledBatchedKernel,
                         public scanner::VideoKernel {
public:
  PatchMatchKernel(const scanner::KernelConfig &config)
      : scanner::StenciledBatchedKernel(config) {}

  template <typename T>
  void writeMatToColumn(const Mat<T> mat, scanner::Elements &col) {
    write_vector_to_column<vector<float>>(col, mat.GetData());
  }

  void execute(const scanner::StenciledBatchedElements &input_cols,
               scanner::BatchedElements &output_cols) override {
    int num_images = input_cols[0][0].size();
    int ref_image_index = 0;
    int ref_image_id =
        read_single_from_element<int>(input_cols[9][0][ref_image_index]);

    std::cout << "reference image id: " << ref_image_id << std::endl;
    std::vector<Image> images;
    std::vector<DepthMap> depth_maps;
    std::vector<NormalMap> normal_maps;
    images.clear();
    for (int i = 0; i < num_images; i++) {
      float *R = readArrayFromElement<float>(input_cols[0][0][i]);
      float *T = readArrayFromElement<float>(input_cols[1][0][i]);

      float *K = readArrayFromElement<float>(input_cols[2][0][i]);
      // float *P = readArrayFromElement<float >(input_cols[3][i]);
      // float *InvP = readArrayFromElement<float >(input_cols[4][i]);
      size_t width = read_single_from_element<size_t>(input_cols[3][0][i]);
      size_t height = read_single_from_element<size_t>(input_cols[4][0][i]);
      size_t scan_width = read_single_from_element<size_t>(input_cols[5][0][i]);

      images.emplace_back("", width, height, K, R, T);
      int bpp = scan_width / width * 8;
      FIBITMAP *fibitmap = FreeImage_ConvertFromRawBits(
          input_cols[6][0][i].buffer, width, height, scan_width, bpp,
          FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);

      colmap::Bitmap bitmap(fibitmap);
      colmap::Bitmap bitmap_grey = bitmap.CloneAsGrey();

      images[i].SetBitmap(bitmap_grey);
    }
    float depth_min =
        read_single_from_element<float>(input_cols[7][0][ref_image_index]);
    float depth_max =
        read_single_from_element<float>(input_cols[8][0][ref_image_index]);

    colmap::mvs::PatchMatchOptions options;
    options.geom_consistency = false;
    options.cache_size = 16;

    // create patch match problem
    PatchMatch::Problem problem;
    problem.ref_image_idx = ref_image_index;
    problem.src_image_idxs.clear();
    problem.src_image_idxs.reserve(num_images - 1);
    for (int i = 0; i < num_images; i++) {
      if (i != ref_image_index) {
        problem.src_image_idxs.push_back(i);
      }
    }
    problem.images = &images;
    problem.depth_maps = &depth_maps;
    problem.normal_maps = &normal_maps;

    // set patch match options
    if (options.depth_min < 0 || options.depth_max < 0) {
      options.depth_min = depth_min;
      options.depth_max = depth_max;
    }

    options.gpu_index = std::to_string(0);

    if (options.sigma_spatial <= 0.0f) {
      options.sigma_spatial = options.window_radius;
    }

    problem.Print();
    options.Print();

    PatchMatch patch_match(options, problem);
    try {
      patch_match.Run();
    } catch (...) {
      std::cerr << "Cannot patch match failed on image " << ref_image_id
                << std::endl;
      return;
    }

    std::cout << "Depth width height: " << patch_match.GetDepthMap().GetWidth()
              << " " << patch_match.GetDepthMap().GetHeight() << std::endl;
    std::cout << "Width height: "
              << read_single_from_element<size_t>(input_cols[5][0][0])
              << std::endl;

    writeMatToColumn<float>(patch_match.GetDepthMap(), output_cols[0]);
    writeMatToColumn<float>(patch_match.GetNormalMap(), output_cols[1]);
  }
};

REGISTER_OP(PatchMatch)
    .stencil()
    .input("R")
    .input("T")
    .input("K")
    .input("width")
    .input("height")
    .input("scan_width")
    .input("bitmap")
    .input("depth_min")
    .input("depth_max")
    .input("image_id")
    .output("depth_map")
    .output("normal_map");

REGISTER_KERNEL(PatchMatch, PatchMatchKernel)
    .device(scanner::DeviceType::CPU)
    .batch()
    .num_devices(1);
