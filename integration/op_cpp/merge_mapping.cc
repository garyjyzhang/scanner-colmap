#include "io.cc"

#include "merge_mapping.pb.h"

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

const double kMaxReprojError = 64.0;

using scanner::Element;

class MergeMappingCPUKernel : public scanner::BatchedKernel,
                              public scanner::VideoKernel {
public:
  MergeMappingCPUKernel(const scanner::KernelConfig &config)
      : scanner::BatchedKernel(config) {
    MergeMappingCPUArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    this->num_models_ = args.num_models();
  }

  void execute(const scanner::BatchedElements &input_cols,
               scanner::BatchedElements &output_cols) override {
    auto &cluster_id_batch = input_cols[0];
    auto &cameras_batch = input_cols[1];
    auto &images_batch = input_cols[2];
    auto &points3d_batch = input_cols[3];

    int first_cluster_id =
        read_single_from_element<int>(cluster_id_batch.front());
    int last_cluster_id =
        read_single_from_element<int>(cluster_id_batch.back());

    Reconstruction merged_reconstruction;
    loadReconstruction(cameras_batch[0], images_batch[0], points3d_batch[0],
                       merged_reconstruction);

    for (int i = 1; i < this->num_models_; i++) {
      int cluster_id = read_single_from_element<int>(cluster_id_batch[i]);
      printf("merging submodel #%d and #%d\n", first_cluster_id, cluster_id);

      Reconstruction reconstruction;
      loadReconstruction(cameras_batch[i], images_batch[i], points3d_batch[i],
                         reconstruction);

      const auto &common_image_ids =
          merged_reconstruction.FindCommonRegImageIds(reconstruction);
      for (const auto &image_id : common_image_ids) {
        printf("common image id: %d\n", image_id);
      }

      if (merged_reconstruction.Merge(reconstruction, kMaxReprojError))
        printf("successfully merged models #%d and #%d\n", first_cluster_id,
               cluster_id);
      else
        printf("failed to merge models #%d and #%d\n", first_cluster_id,
               cluster_id);
    }

    std::string save_dir =
        colmap::StringPrintf("%d_%d", first_cluster_id, last_cluster_id);
    colmap::CreateDirIfNotExists(save_dir);
    merged_reconstruction.Write(save_dir);

    write_single_to_column<int>(output_cols[0], first_cluster_id);
    write_reconstruction_to_columns(output_cols[1], output_cols[2],
                                    output_cols[3], save_dir);
  }

private:
  int num_models_;
};

REGISTER_OP(MergeMappingCPU)
    .input("cluster_id")
    .input("cameras")
    .input("images")
    .input("points3d")
    .output("cluster_id")
    .output("cameras")
    .output("images")
    .output("points3D")
    .protobuf_name("MergeMappingCPUArgs");

REGISTER_KERNEL(MergeMappingCPU, MergeMappingCPUKernel)
    .device(scanner::DeviceType::CPU)
    .batch()
    .num_devices(1);
