#include "io.cc"

#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/common.h"
#include "scanner/util/memory.h"

#include <colmap/base/database.h>
#include <colmap/controllers/incremental_mapper.h>
#include <colmap/feature/types.h>
#include <colmap/feature/utils.h>
#include <colmap/util/misc.h>

#include <stdio.h>

using colmap::BundleAdjustmentOptions;
using colmap::Camera;
using colmap::camera_t;
using colmap::CorrespondenceGraph;
using colmap::Database;
using colmap::DatabaseCache;
using colmap::Image;
using colmap::image_t;
using colmap::IncrementalMapper;
using colmap::IncrementalMapperOptions;
using colmap::ParallelBundleAdjuster;
using colmap::Reconstruction;
using colmap::StringPrintf;
using scanner::Elements;

typedef vector<image_t> PairImageIds;
typedef vector<TwoViewGeometry> TwoViewGeometries;

// temporary database storage
const static std::string kTempDatabasePath = ".db";

class IncrementalMappingCPUKernel : public scanner::StenciledBatchedKernel,
                                    public scanner::VideoKernel {
public:
  IncrementalMappingCPUKernel(const scanner::KernelConfig &config)
      : scanner::StenciledBatchedKernel(config) {}

  size_t TriangulateImage(const IncrementalMapperOptions &options,
                          const Image &image, IncrementalMapper *mapper) {
    // std::cout << "  => Continued observations: " << image.NumPoints3D()
    //           << std::endl;
    const size_t num_tris =
        mapper->TriangulateImage(options.Triangulation(), image.ImageId());
    // std::cout << "  => Added observations: " << num_tris << std::endl;
    return num_tris;
  }

  void AdjustGlobalBundle(const IncrementalMapperOptions &options,
                          IncrementalMapper *mapper) {
    BundleAdjustmentOptions custom_options = options.GlobalBundleAdjustment();

    const size_t num_reg_images = mapper->GetReconstruction().NumRegImages();

    // Use stricter convergence criteria for first registered images.
    const size_t kMinNumRegImages = 10;
    if (num_reg_images < kMinNumRegImages) {
      custom_options.solver_options.function_tolerance /= 10;
      custom_options.solver_options.gradient_tolerance /= 10;
      custom_options.solver_options.parameter_tolerance /= 10;
      custom_options.solver_options.max_num_iterations *= 2;
      custom_options.solver_options.max_linear_solver_iterations = 200;
    }

    // printf("Global bundle adjustment");
    if (options.ba_global_use_pba && num_reg_images >= kMinNumRegImages &&
        ParallelBundleAdjuster::IsSupported(custom_options,
                                            mapper->GetReconstruction())) {
      mapper->AdjustParallelGlobalBundle(
          custom_options, options.ParallelGlobalBundleAdjustment());
    } else {
      mapper->AdjustGlobalBundle(custom_options);
    }
  }

  void IterativeLocalRefinement(const IncrementalMapperOptions &options,
                                const image_t image_id,
                                IncrementalMapper *mapper) {
    auto ba_options = options.LocalBundleAdjustment();
    for (int i = 0; i < options.ba_local_max_refinements; ++i) {
      const auto report = mapper->AdjustLocalBundle(
          options.Mapper(), ba_options, options.Triangulation(), image_id,
          mapper->GetModifiedPoints3D());
      // std::cout << "  => Merged observations: "
      //           << report.num_merged_observations << std::endl;
      // std::cout << "  => Completed observations: "
      //           << report.num_completed_observations << std::endl;
      // std::cout << "  => Filtered observations: "
      //           << report.num_filtered_observations << std::endl;
      const double changed =
          (report.num_merged_observations + report.num_completed_observations +
           report.num_filtered_observations) /
          static_cast<double>(report.num_adjusted_observations);
      // std::cout << StringPrintf("  => Changed observations: %.6f", changed)
      //           << std::endl;
      if (changed < options.ba_local_max_refinement_change) {
        break;
      }
      // Only use robust cost function for first iteration.
      ba_options.loss_function_type =
          BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
    }
    mapper->ClearModifiedPoints3D();
  }

  void IterativeGlobalRefinement(const IncrementalMapperOptions &options,
                                 IncrementalMapper *mapper) {
    // printf("Retriangulation");
    CompleteAndMergeTracks(options, mapper);
    // std::cout << "  => Retriangulated observations: "
    //           << mapper->Retriangulate(options.Triangulation()) << std::endl;

    for (int i = 0; i < options.ba_global_max_refinements; ++i) {
      const size_t num_observations =
          mapper->GetReconstruction().ComputeNumObservations();
      size_t num_changed_observations = 0;
      AdjustGlobalBundle(options, mapper);
      num_changed_observations += CompleteAndMergeTracks(options, mapper);
      num_changed_observations += FilterPoints(options, mapper);
      const double changed =
          static_cast<double>(num_changed_observations) / num_observations;
      // std::cout << StringPrintf("  => Changed observations: %.6f", changed)
      //           << std::endl;
      if (changed < options.ba_global_max_refinement_change) {
        break;
      }
    }

    FilterImages(options, mapper);
  }

  void ExtractColors(const std::string &image_path, const image_t image_id,
                     Reconstruction *reconstruction) {
    if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
      std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                                reconstruction->Image(image_id).Name().c_str(),
                                image_path.c_str())
                << std::endl;
    }
  }

  size_t FilterPoints(const IncrementalMapperOptions &options,
                      IncrementalMapper *mapper) {
    const size_t num_filtered_observations =
        mapper->FilterPoints(options.Mapper());
    // std::cout << "  => Filtered observations: " << num_filtered_observations
    //           << std::endl;
    return num_filtered_observations;
  }

  size_t FilterImages(const IncrementalMapperOptions &options,
                      IncrementalMapper *mapper) {
    const size_t num_filtered_images = mapper->FilterImages(options.Mapper());
    // std::cout << "  => Filtered images: " << num_filtered_images <<
    // std::endl;
    return num_filtered_images;
  }

  size_t CompleteAndMergeTracks(const IncrementalMapperOptions &options,
                                IncrementalMapper *mapper) {
    const size_t num_completed_observations =
        mapper->CompleteTracks(options.Triangulation());
    // std::cout << "  => Merged observations: " << num_completed_observations
    //           << std::endl;
    const size_t num_merged_observations =
        mapper->MergeTracks(options.Triangulation());
    // std::cout << "  => Completed observations: " << num_merged_observations
    //           << std::endl;
    return num_completed_observations + num_merged_observations;
  }

  Image createImage(image_t id, camera_t camera_id,
                    FeatureKeypoints &keypoints) {
    colmap::Image image;
    image.SetName("image_" + std::to_string(id));
    image.SetImageId(id);
    image.SetCameraId(camera_id);

    const std::vector<Eigen::Vector2d> points =
        colmap::FeatureKeypointsToPointsVector(keypoints);
    image.SetPoints2D(points);

    return image;
  }

  void LoadDatabase(Database &db, const Elements &image_id_stencil,
                    const Elements &pair_image_id_stencil,
                    const Elements &two_view_geometry_stencil,
                    const Elements &keypoints_stencil,
                    const Elements &camera_stencil, int num_pivot_images) {
    ////////////////////////////////////////////////////////////////////////////
    // load all the resources from stencils
    ////////////////////////////////////////////////////////////////////////////
    std::cout << "Loading from stencils to database" << std::endl;
    int stencil_size = image_id_stencil.size();
    std::vector<image_t> pivot_image_ids;

    for (int i = 0; i < image_id_stencil.size(); i++) {
      // load image id
      image_t image_id = read_single_from_element<image_t>(image_id_stencil[i]);

      // the first num_pivot_images are pivots
      if (i < num_pivot_images) {
        pivot_image_ids.push_back(image_id);
      }

      // load camera
      colmap::Camera camera;
      read_camera_from_element(camera_stencil[i], &camera);
      if (!db.ExistsCamera(camera.CameraId())) {
        db.WriteCamera(camera, true);
        printf("Loaded camera #%d into db\n", camera.CameraId());
      } else {
        printf("Camera #%d already exists in db\n", camera.CameraId());
      }

      if (!db.ExistsImage(image_id)) {
        // load keypoint
        auto keypoints = read_keypoints_from_element(keypoints_stencil[i]);
        // create image
        Image image = createImage(image_id, camera.CameraId(), keypoints);

        db.WriteImage(image, true);
        db.WriteKeypoints(image_id, keypoints);
        printf("Loaded image #%d into db\n", image_id);
      } else {
        printf("Image #%d already exists in db\n", image_id);
      }
    }

    for (int i = 0; i < num_pivot_images; i++) {
      image_t pivot_image_id = pivot_image_ids[i];
      std::cout << "unpacking matches for pivot image: " << pivot_image_id
                << std::endl;

      // load pairs
      PairImageIds pair_image_ids =
          read_vector_from_element<PairImageIds>(pair_image_id_stencil[i]);
      printf("Read %ld pair ids\n", pair_image_ids.size());

      // load two view geometries
      vector<TwoViewGeometry> two_view_geometries =
          read_two_view_geometries(two_view_geometry_stencil[i]);
      printf("Read %ld tvgs\n", two_view_geometries.size());

      // number of pairs must be equal to number of tvgs, since they are 1-to-1
      CHECK_EQ(pair_image_ids.size(), two_view_geometries.size());

      for (int pair_index = 0; pair_index < pair_image_ids.size();
           pair_index++) {
        db.WriteTwoViewGeometry(pivot_image_id, pair_image_ids[pair_index],
                                two_view_geometries[pair_index]);
      }
    }

    std::cout << "Loaded cluster data into db" << std::endl;
  }

  // converted from colmap::IncrementalMapperController::Reconstruct
  void execute(const scanner::StenciledBatchedElements &input_cols,
               scanner::BatchedElements &output_cols) override {
    auto &image_id_stencil = input_cols[0][0];
    auto &pair_image_id_stencil = input_cols[1][0];
    auto &two_view_geometry_stencil = input_cols[2][0];
    auto &keypoints_stencil = input_cols[3][0];
    auto &camera_stencil = input_cols[4][0];

    CHECK_GT(image_id_stencil.size(), 0);

    // the number of pivot images is equal to the batch size
    int num_pivot_images = 10;
    int batch_size = input_cols[0].size();
    std::cout << "num pivot images: " << num_pivot_images << std::endl;
    // the id of the cluster is caculated by first_image_id / cluster size
    // which should be unique if the batch size is consistent
    int cluster_id = read_single_from_element<size_t>(image_id_stencil[0]);
    std::string tmp_db_path = std::to_string(cluster_id) + kTempDatabasePath;

    IncrementalMapperOptions options;

    remove(tmp_db_path.c_str());
    Database db(tmp_db_path);
    LoadDatabase(db, image_id_stencil, pair_image_id_stencil,
                 two_view_geometry_stencil, keypoints_stencil, camera_stencil,
                 num_pivot_images);

    DatabaseCache cache;
    cache.Load(db, options.min_num_matches, options.ignore_watermarks,
               options.image_names);
    remove(tmp_db_path.c_str());
    IncrementalMapper mapper(&cache);

    IncrementalMapper::Options mapper_options = options.Mapper();

    Reconstruction reconstruction;
    mapper.BeginReconstruction(&reconstruction);

    ////////////////////////////////////////////////////////////////////////////
    // Register initial pair
    ////////////////////////////////////////////////////////////////////////////
    image_t image_id1 = colmap::kInvalidImageId,
            image_id2 = colmap::kInvalidImageId;

    CHECK(mapper.FindInitialImagePair(mapper_options, &image_id1, &image_id2));
    printf("Initializing with image pair %d and %d\n", image_id1, image_id2);

    CHECK(
        mapper.RegisterInitialImagePair(mapper_options, image_id1, image_id2));
    printf("Registered image pair %d and %d\n", image_id1, image_id2);

    ////////////////////////////////////////////////////////////////////////////
    // Register all other images in the cluster
    ////////////////////////////////////////////////////////////////////////////
    size_t ba_prev_num_reg_images = reconstruction.NumRegImages();
    size_t ba_prev_num_points = reconstruction.NumPoints3D();
    bool reg_next_success = true;

    while (reg_next_success) {
      const std::vector<image_t> next_image_ids =
          mapper.FindNextImages(mapper_options);
      if (next_image_ids.empty())
        break;

      for (const auto next_image_id : next_image_ids) {
        if (!mapper.RegisterNextImage(mapper_options, next_image_id)) {
          continue;
        };
        printf("Registered %ld images\n", mapper.NumTotalRegImages());
        printf("Image in cache not registered: %ld",
               cache.NumImages() - mapper.NumTotalRegImages());
        const Image &next_image = reconstruction.Image(next_image_id);
        TriangulateImage(options, next_image, &mapper);
        IterativeLocalRefinement(options, next_image_id, &mapper);

        if (reconstruction.NumRegImages() >=
                options.ba_global_images_ratio * ba_prev_num_reg_images ||
            reconstruction.NumRegImages() >=
                options.ba_global_images_freq + ba_prev_num_reg_images ||
            reconstruction.NumPoints3D() >=
                options.ba_global_points_ratio * ba_prev_num_points ||
            reconstruction.NumPoints3D() >=
                options.ba_global_points_freq + ba_prev_num_points) {
          IterativeGlobalRefinement(options, &mapper);
          ba_prev_num_points = reconstruction.NumPoints3D();
          ba_prev_num_reg_images = reconstruction.NumRegImages();
        }
      }

      IterativeGlobalRefinement(options, &mapper);
    }

    // run a final global ba if the last incremental ba is not global
    if (reconstruction.NumRegImages() >= 2 &&
        reconstruction.NumRegImages() != ba_prev_num_reg_images &&
        reconstruction.NumPoints3D() != ba_prev_num_points) {
      IterativeGlobalRefinement(options, &mapper);
    }

    mapper.EndReconstruction(false);
    printf("submodel #%d reconstruction complete\n", cluster_id);

    std::string save_dir = std::to_string(cluster_id);
    colmap::CreateDirIfNotExists(save_dir);
    reconstruction.Write(save_dir);

    write_single_and_fill_column<int>(output_cols[0], cluster_id, batch_size);
    write_reconstruction_to_columns(output_cols[1], output_cols[2],
                                    output_cols[3], save_dir, batch_size);
  }
};

REGISTER_OP(IncrementalMappingCPU)
    .stencil()
    .input("image_id")
    .input("pair_image_ids")
    .input("two_view_geometries")
    .input("keypoints")
    .input("camera")
    .output("cluster_id")
    .output("cameras_bin")
    .output("images_bin")
    .output("points3D_bin");

REGISTER_KERNEL(IncrementalMappingCPU, IncrementalMappingCPUKernel)
    .device(scanner::DeviceType::CPU)
    .batch()
    .num_devices(1);
