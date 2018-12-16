#include "io.cc"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"

#include <colmap/base/undistort.h>
#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>

using colmap::image_t;
using colmap::Bitmap;
using colmap::Image;

class ImageUndistorter : public scanner::StenciledBatchedKernel,
                         public scanner::VideoKernel {

  void execute(const scanner::StenciledBatchedElements &input_cols,
               scanner::BatchedElements &output_cols) override {
    loadReconstruction()

        const image_t image_id =
            reconstruction_.RegImageIds().at(reg_image_idx);
    const Image &image = reconstruction_.Image(image_id);
    const Camera &camera = reconstruction_.Camera(image.CameraId());

    const std::string output_image_path =
        JoinPaths(output_path_, "images", image.Name());

    Bitmap distorted_bitmap;
    const std::string input_image_path = JoinPaths(image_path_, image.Name());
    if (!distorted_bitmap.Read(input_image_path)) {
      std::cerr << "ERROR: Cannot read image at path " << input_image_path
                << std::endl;
      return;
    }

    Bitmap undistorted_bitmap;
    Camera undistorted_camera;
    UndistortCameraOptions options;
    colmap::UndistortImage(options, distorted_bitmap, camera,
                           &undistorted_bitmap, &undistorted_camera);

    undistorted_bitmap.Write(output_image_path);
  }
}

REGISTER_OP(IncrementalMappingCPU)
    .stencil()
    .input("cameras_bin")
    .input("images_bin")
    .input("points3D_bin")
    .protobuf_name("IncrementalMappingCPUArgs");
;

REGISTER_KERNEL(IncrementalMappingCPU, IncrementalMappingCPUKernel)
    .device(scanner::DeviceType::CPU)
    .batch()
    .num_devices(1);
