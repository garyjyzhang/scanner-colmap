#include "io.cc"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"
#include "scanner/util/memory.h"

// A simple kernel to loop over the images and assign id's
class PrepareImageKernel : public scanner::Kernel, public scanner::VideoKernel {
public:
  PrepareImageKernel(const scanner::KernelConfig &config)
      : scanner::Kernel(config), id_counter_(0) {}

  void execute(const scanner::Elements &input_cols,
               scanner::Elements &output_cols) override {

    writeSingleToElement<size_t>(output_cols[0], id_counter_);
    id_counter_++;
  }

private:
  size_t id_counter_;
};

REGISTER_OP(PrepareImage).frame_input("frames").output("image_ids");

REGISTER_KERNEL(PrepareImage, PrepareImageKernel)
    .device(scanner::DeviceType::CPU)
    .num_devices(1);
