#include "io.cc"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"
#include "scanner/util/memory.h"

#include <colmap/base/reconstruction.h>
#include <colmap/base/similarity_transform.h>
#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>
#include <colmap/util/misc.h>
#include <colmap/util/string.h>

const double kMaxReprojError = 256.0;

using scanner::Element;

class MergeMappingCPUKernel : public scanner::BatchedKernel,
                              public scanner::VideoKernel {
public:
  MergeMappingCPUKernel(const scanner::KernelConfig &config)
      : scanner::BatchedKernel(config) {}

  void execute(const scanner::BatchedElements &input_cols,
               scanner::BatchedElements &output_cols) override {
    auto &cluster_id_batch = input_cols[0];
    auto &cameras_batch = input_cols[1];
    auto &images_batch = input_cols[2];
    auto &points3d_batch = input_cols[3];

    int num_models = input_cols[0].size();

    // caluclate range of leaf clusters being merged
    int initial_model = 0;
    int num_reconstructions = 5;
    int first_front_id =
        readSingleFromElement<int>(cluster_id_batch[initial_model]);
    int last_front_id = readSingleFromElement<int>(
        cluster_id_batch[initial_model + num_reconstructions]);

    Reconstruction merged_reconstruction;
    loadReconstruction(cameras_batch[initial_model],
                       images_batch[initial_model],
                       points3d_batch[initial_model], merged_reconstruction);

    for (const auto &image_id : merged_reconstruction.RegImageIds()) {
      printf("submodel #%d has registered image: %d\n", initial_model,
             image_id);
    }

    for (int i = initial_model + 1; i <= initial_model + num_reconstructions;
         i++) {
      Reconstruction reconstruction;
      loadReconstruction(cameras_batch[i], images_batch[i], points3d_batch[i],
                         reconstruction);
      std::cout << "loaded submodel #" << i << std::endl;

      for (const auto &image_id : reconstruction.RegImageIds()) {
        printf("submodel #%d has registered image: %d\n", i, image_id);
      }

      Eigen::Matrix3x4d alignment;

      const auto &common_image_ids =
          merged_reconstruction.FindCommonRegImageIds(reconstruction);
      for (const auto &image_id : common_image_ids) {
        printf("common image id: %d\n", image_id);
      }

      // if (!colmap::ComputeAlignmentBetweenReconstructions(
      //         reconstruction, merged_reconstruction, 0.3, kMaxReprojError,
      //         &alignment)) {
      //   std::cout << "no alignment!" << std::endl;
      // }
      if (merged_reconstruction.Merge(reconstruction, kMaxReprojError))
        printf("merged models #%d and #%d\n", 0, i);
    }

    std::string save_dir =
        colmap::StringPrintf("%d_%d", first_front_id, last_front_id);
    colmap::CreateDirIfNotExists(save_dir);
    merged_reconstruction.Write(save_dir);

    writeSingleAndFillColumn<int>(output_cols[0], first_front_id, num_models);
    writeReconstructionToColumns(output_cols[1], output_cols[2], output_cols[3],
                                 save_dir, num_models);
  }
};

REGISTER_OP(MergeMappingCPU)
    .input("cluster_id")
    .input("cameras")
    .input("images")
    .input("points3d")
    .output("cluster_id")
    .output("cameras")
    .output("images")
    .output("points3D");

REGISTER_KERNEL(MergeMappingCPU, MergeMappingCPUKernel)
    .device(scanner::DeviceType::CPU)
    .batch()
    .num_devices(1);
