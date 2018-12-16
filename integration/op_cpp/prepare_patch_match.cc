#include "io.cc"

#include "prepare_patch_match.pb.h"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"
#include "scanner/util/cuda.h"
#include "scanner/util/memory.h"

#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>
#include <colmap/mvs/model.h>
#include <colmap/util/bitmap.h>

using colmap::image_t;
using colmap::Bitmap;
using colmap::Image;

// Based on colmap::PatchMatchController
class PreparePatchMatch : public scanner::BatchedKernel,
                          public scanner::VideoKernel {
public:
  PreparePatchMatch(const scanner::KernelConfig &config)
      : scanner::BatchedKernel(config) {
    PreparePatchMatchArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    model_.ReadFromCOLMAP(args.sparse_reconstruction_path());

    // set_device();
    device_ = scanner::CPU_DEVICE;
  }

  void execute(const scanner::BatchedElements &input_cols,
               scanner::BatchedElements &output_cols) override {
    const auto &depth_range_list = model_.ComputeDepthRanges();

    std::cout << "model images: " << model_.images.size() << std::endl;
    // based on PatchMatchController::ReadProblems()
    for (int i = 0; i < model_.images.size(); i++) {
      colmap::mvs::Image image = model_.images.at(i);
      Bitmap bitmap;
      bitmap.Read(image.GetPath(), true);
      bitmap.Rescale(image.GetWidth(), image.GetHeight());
      image.SetBitmap(bitmap);

      writeArrayToColumn<float>(output_cols[0], image.GetR(), 9, device_);
      writeArrayToColumn<float>(output_cols[1], image.GetT(), 3, device_);
      writeArrayToColumn<float>(output_cols[2], image.GetK(), 9, device_);

      size_t width = image.GetWidth();
      size_t height = image.GetHeight();
      write_single_to_column<size_t>(output_cols[3], width, device_);
      write_single_to_column<size_t>(output_cols[4], height, device_);

      const FIBITMAP *fibitmap = image.GetBitmap().Data();
      size_t scan_width = FreeImage_GetPitch(const_cast<FIBITMAP *>(fibitmap));
      int bpp = scan_width / width * 8;
      write_single_to_column<size_t>(output_cols[5], scan_width, device_);
      std::cout << "pitch bpp: " << scan_width << " " << bpp << std::endl;

      // store bitmap
      u8 *frame_buffer = new_buffer(device_, height * scan_width);
      FreeImage_ConvertToRawBits(frame_buffer, const_cast<FIBITMAP *>(fibitmap),
                                 scan_width, bpp, FI_RGBA_RED_MASK,
                                 FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);
      // could also store as frame column, but storing as blob for easy
      // conversion later on
      insert_element(output_cols[6], frame_buffer, height * scan_width);

      write_single_to_column<float>(output_cols[7],
                                    depth_range_list.at(i).first, device_);

      write_single_to_column<float>(output_cols[8],
                                    depth_range_list.at(i).second, device_);
      write_single_to_column<int>(output_cols[9], i, device_);
    }
  }

private:
  void set_device() {
    if (device_.type == scanner::DeviceType::GPU) {
      CUDA_PROTECT({ CU_CHECK(cudaSetDevice(device_.id)); });
    }
  }

  colmap::mvs::Model model_;
  scanner::DeviceHandle device_;
};

REGISTER_OP(PreparePatchMatch)
    .input("image_id")
    .output("R")
    .output("T")
    .output("K")
    .output("width")
    .output("height")
    .output("scan_width")
    .output("bitmap")
    .output("depth_min")
    .output("depth_max")
    .output("image_id")
    .protobuf_name("PreparePatchMatchArgs");
;

REGISTER_KERNEL(PreparePatchMatch, PreparePatchMatch)
    .device(scanner::DeviceType::CPU)
    .batch()
    .num_devices(1);
