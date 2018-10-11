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

using scanner::u8;
using std::size_t;
class SiftExtractionKernel : public scanner::Kernel,
                             public scanner::VideoKernel {
public:
  SiftExtractionKernel(const scanner::KernelConfig &config)
      : scanner::Kernel(config), id_counter(0) {}

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

  void createKeypointsBuffer(colmap::FeatureKeypoints &keypoints,
                             scanner::Element &output) {
    size_t rows = keypoints.size();
    size_t keypointByteSize = sizeof(colmap::FeatureKeypoint);
    size_t keypointsByteSize = keypointByteSize * rows;
    u8 *keypointsBuffer =
        new_buffer(scanner::CPU_DEVICE, keypointsByteSize + sizeof(rows));

    std::memcpy(keypointsBuffer, &rows, sizeof(rows));
    std::memcpy(keypointsBuffer + sizeof(rows), keypoints.data(),
                keypointsByteSize);

    insert_element(output, keypointsBuffer, keypointsByteSize + sizeof(rows));
  }

  void printKeypoint(colmap::FeatureKeypoint &point) {
    printf("point: %f %f %f %f %f %f \n", point.x, point.y, point.a11,
           point.a12, point.a21, point.a22);
  }

  void execute(const scanner::Elements &input_cols,
               scanner::Elements &output_cols) override {
    auto &frame_col = input_cols[0];

    check_frame(scanner::CPU_DEVICE, frame_col);

    const scanner::Frame *frame = frame_col.as_const_frame();

    std::cout << frame->width() << ' ' << frame->height() << ' '
              << frame->channels() << std::endl;

    int bpp = 24; // assume 3 color channels
    int pitch = frame->width() * 3;

    FIBITMAP *fibitmap = FreeImage_ConvertFromRawBits(
        frame->data, frame->width(), frame->height(), pitch, bpp,
        FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);

    colmap::Bitmap bitmap(fibitmap);
    colmap::Bitmap bitmap_grey = bitmap.CloneAsGrey();
    colmap::SiftExtractionOptions siftOptions;
    siftOptions.use_gpu = false;

    colmap::FeatureKeypoints keypoints;
    colmap::FeatureDescriptors descriptors;
    std::cout << "starting extraction..." << std::endl;

    // debug use image_reader for image reading
    // colmap::ImageReaderOptions image_reader_opt;
    // image_reader_opt.image_path = "/app/gerrard-hall/images/";
    // colmap::Database database("/app/integration/colmap.db");
    // colmap::ImageReader image_reader(image_reader_opt, &database);
    // colmap::Camera camera;
    // colmap::Image image;
    // colmap::Bitmap read_bitmap;
    // image_reader.Next(&camera, &image, &read_bitmap);
    // std::cout << "image reader read: " << image.Name() << std::endl;
    // debug end

    resizeBitmap(bitmap_grey, siftOptions.max_image_size);

    // colmap::ExtractSiftFeaturesCPU(siftOptions, read_bitmap, &keypoints,
    // &descriptors);
    colmap::ExtractSiftFeaturesCPU(siftOptions, bitmap_grey, &keypoints,
                                   &descriptors);

    std::cout << "extraction complete" << std::endl;

    // debug print key point x y, #features
    // for(auto keypoint: keypoints) {
    //   std::cout << "x: " << keypoint.x << "y: " << keypoint.y << std::endl;
    // }
    // std::cout << "number of features: " << keypoints.size() << std::endl;

    // Create buffer for keypoints
    createKeypointsBuffer(keypoints, output_cols[1]);
    printKeypoint(keypoints.front());
    printKeypoint(keypoints.back());

    // Create buffer for descriptors
    size_t rows = descriptors.rows();
    size_t cols = descriptors.cols();
    size_t index_size = sizeof(size_t);
    size_t matrix_num_bytes =
        descriptors.size() *
        sizeof(typename colmap::FeatureDescriptors::Scalar);

    size_t descriptorsByteSize = matrix_num_bytes + index_size * 2;
    u8 *descriptorsBuffer =
        new_buffer(scanner::CPU_DEVICE, descriptorsByteSize);
    std::memcpy(descriptorsBuffer, &rows, index_size);
    std::memcpy(descriptorsBuffer + index_size, &cols, index_size);
    std::memcpy(descriptorsBuffer + index_size * 2, descriptors.data(),
                descriptors.size());
    std::cout << "image id: " << id_counter << std::endl;
    std::cout << "rows cols: " << descriptors.rows() << " "
              << descriptors.cols() << std::endl;
    std::cout << "matrix size: " << descriptors.size() << std::endl;

    // for debug image output
    // u8* bits = new_buffer(scanner::CPU_DEVICE, bitmap_grey.Width() *
    // bitmap_grey.Height()); FIBITMAP* gray = (FIBITMAP*)bitmap_grey.Data();
    // scanner::FrameInfo output_frame_info(bitmap_grey.Height(),
    // bitmap_grey.Width(), bitmap_grey.Channels(), scanner::FrameType::U8);
    // FreeImage_ConvertToRawBits(bits, gray, bitmap_grey.Width(), 8,
    // FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);
    // scanner::Frame* output_frame = new scanner::Frame(output_frame_info,
    // bits);

    // insert_frame(output_cols[0], output_frame);

    u8 *image_id_buffer = new_buffer(scanner::CPU_DEVICE, sizeof(int));
    std::memcpy(image_id_buffer, &id_counter, sizeof(int));

    insert_element(output_cols[0], image_id_buffer, sizeof(int));
    insert_element(output_cols[2], descriptorsBuffer, descriptorsByteSize);

    id_counter++;
  }

private:
  int id_counter;
};

REGISTER_OP(SiftExtraction)
    .frame_input("frame")
    .output("image_id")
    .output("keypoints")
    .output("descriptors")
    .protobuf_name("siftExtractionArgs");

REGISTER_KERNEL(SiftExtraction, SiftExtractionKernel)
    .device(scanner::DeviceType::CPU)
    .num_devices(1);
