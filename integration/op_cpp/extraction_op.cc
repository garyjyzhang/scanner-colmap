#include <string>

#include "io.cc"
#include "siftExtraction.pb.h"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"
#include "scanner/util/memory.h"

#include <colmap/base/image_reader.h>
#include <colmap/feature/sift.h>
#include <colmap/feature/types.h>
#include <colmap/util/bitmap.h>
#include <colmap/util/option_manager.h>

using colmap::Bitmap;
using colmap::Camera;
using colmap::Image;

// Kernel class to perform colmap's SIFT extraction
class SiftExtractionKernel : public scanner::Kernel,
                             public scanner::VideoKernel {
public:
  SiftExtractionKernel(const scanner::KernelConfig &config)
      : scanner::Kernel(config) {}

  void resizeBitmap(colmap::Bitmap &bitmap, int max_image_size) {
    if (static_cast<int>(bitmap.Width()) > max_image_size ||
        static_cast<int>(bitmap.Height()) > max_image_size) {
      // Fit the down-sampled version exactly into the max dimensions.
      const double scale = static_cast<double>(max_image_size) /
                           std::max(bitmap.Width(), bitmap.Height());
      const int new_width = static_cast<int>(bitmap.Width() * scale);
      const int new_height = static_cast<int>(bitmap.Height() * scale);

      bitmap.Rescale(new_width, new_height);
    }
  }

  // TODO: pass in image reader options
  // converted from colmap::ImageReader::Next
  void ReadImage(size_t image_id, Camera *camera, Image *image,
                 Bitmap *bitmap) {
    // use default options for now
    colmap::ImageReaderOptions options;

    image->SetImageId(image_id);
    // use the format image_${image_id} for name
    image->SetName("image_" + std::to_string(image_id));

    // extract camera model
    double focal_length = 0.0;
    if (bitmap->ExifFocalLength(&focal_length)) {
      camera->SetPriorFocalLength(true);
    } else {
      focal_length = options.default_focal_length_factor *
                     std::max(bitmap->Width(), bitmap->Height());
      camera->SetPriorFocalLength(false);
    }

    camera->SetModelIdFromName(options.camera_model);
    camera->InitializeWithId(camera->ModelId(), focal_length, bitmap->Width(),
                             bitmap->Height());
    camera->SetWidth(static_cast<size_t>(bitmap->Width()));
    camera->SetHeight(static_cast<size_t>(bitmap->Height()));

    // if (!camera->VerifyParams()) {
    //   return Status::CAMERA_PARAM_ERROR;
    // }

    camera->SetCameraId(image_id);
    image->SetCameraId(camera->CameraId());

    // extract GPS data
    if (!bitmap->ExifLatitude(&image->TvecPrior(0)) ||
        !bitmap->ExifLongitude(&image->TvecPrior(1)) ||
        !bitmap->ExifAltitude(&image->TvecPrior(2))) {
      image->TvecPrior().setConstant(std::numeric_limits<double>::quiet_NaN());
    }
  }

  // void printKeypoint(colmap::FeatureKeypoint &point) {
  //   printf("point: %f %f %f %f %f %f \n", point.x, point.y, point.a11,
  //          point.a12, point.a21, point.a22);
  // }

  void execute(const scanner::Elements &input_cols,
               scanner::Elements &output_cols) override {
    auto &image_id_col = input_cols[0];
    auto &frame_col = input_cols[1];

    size_t image_id = readSingleFromElement<size_t>(image_id_col);

    check_frame(scanner::CPU_DEVICE, frame_col);

    const scanner::Frame *frame = frame_col.as_const_frame();

    // convert Scanner::Frame to colmap::Bitmap
    int bpp = frame->channels() * 8;
    int pitch = frame->width() * frame->channels();

    FIBITMAP *fibitmap = FreeImage_ConvertFromRawBits(
        frame->data, frame->width(), frame->height(), pitch, bpp,
        FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);

    colmap::Bitmap bitmap(fibitmap);
    // only need grey scale for colmap
    colmap::Bitmap bitmap_grey = bitmap.CloneAsGrey();

    // set up configs for SIFT
    // TODO: pass options from kernel configs
    colmap::SiftExtractionOptions siftOptions;
    siftOptions.use_gpu = false; // no gpu for now

    // scale down the image if needed
    resizeBitmap(bitmap_grey, siftOptions.max_image_size);

    // start SIFT extraction process
    std::cout << "starting extraction..." << std::endl;
    colmap::FeatureKeypoints keypoints;
    colmap::FeatureDescriptors descriptors;

    colmap::ExtractSiftFeaturesCPU(siftOptions, bitmap_grey, &keypoints,
                                   &descriptors);

    std::cout << "extraction complete" << std::endl;
    // SIFT extraction end

    // Create Image and Camera
    // Creating a separate camera for each image due to lack of db
    Camera camera;
    Image image;
    ReadImage(image_id, &camera, &image, &bitmap_grey);

    // write to output columns
    writeSingleToElement(output_cols[0], image);
    writeVectorToElement(output_cols[1], keypoints);
    writeMatrixToElement(output_cols[2], descriptors);
    writeSingleToElement(output_cols[3], camera);
  }
};

REGISTER_OP(SiftExtraction)
    .input("image_ids")
    .frame_input("frames")
    .output("colmap_images")
    .output("keypoints")
    .output("descriptors")
    .output("cameras")
    .protobuf_name("siftExtractionArgs");

REGISTER_KERNEL(SiftExtraction, SiftExtractionKernel)
    .device(scanner::DeviceType::CPU)
    .num_devices(1);
