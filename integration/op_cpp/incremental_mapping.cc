#include "mapping_cluster_cache.cc"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"
#include "scanner/util/memory.h"

#include <colmap/base/database.h>
#include <colmap/controllers/incremental_mapper.h>
#include <colmap/feature/matching.h>
#include <colmap/feature/sift.h>
#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>

class IncrementalMappingCPUKernel : public scanner::StenciledBatchedKernel,
                                    public scanner::VideoKernel {
public:
  IncrementalMappingCPUKernel(const scanner::KernelConfig &config)
      : scanner::StenciledBatchedKernel(config) {}

  void execute(const scanner::StenciledBatchedElements &input_cols,
               scanner::BatchedElements &output_cols) override {
    auto &image_id_stencil = input_cols[0][0];
    auto &pair_image_id_stencil = input_cols[1][0];
    auto &two_view_geometry_stencil = input_cols[2][0];
    auto &keypoints_stencil = input_cols[3][0];
    auto &camera_stencil = input_cols[4][0];
    std::cout << "enter" << std::endl;

    // the number of pivot images is equal to the batch size
    int num_pivot_images = input_cols[0].size();

    colmap::IncrementalMapperOptions options;
    MappingClusterCache cache;
    cache.LoadFromStencils(image_id_stencil, pair_image_id_stencil,
                           two_view_geometry_stencil, keypoints_stencil,
                           camera_stencil, num_pivot_images,
                           options.min_num_matches, options.ignore_watermarks);
  }
};

REGISTER_OP(IncrementalMappingCPU)
    .stencil()
    .input("image_id")
    .input("pair_image_ids")
    .input("two_view_geometries")
    .input("keypoints")
    .input("camera")
    .output("two_view_geometry");

REGISTER_KERNEL(IncrementalMappingCPU, IncrementalMappingCPUKernel)
    .device(scanner::DeviceType::CPU)
    .batch()
    .num_devices(1);
