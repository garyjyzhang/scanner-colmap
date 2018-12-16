cmake_minimum_required(VERSION 3.5.2 FATAL_ERROR)

project(ScannerColmap)

find_package(COLMAP REQUIRED)
# or to require a specific version: find_package(COLMAP 3.4 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Op.cmake should be included by a CMake script that will build a custom Scanner
# op. It sets a few default flags and exposes a function build_op for simplifying
# the build process. See examples/tutorial/04_custom_op.py for an example usage.

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/Modules")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
if(NOT COMPILER_SUPPORTS_CXX1Y)
  message(FATAL_ERROR
    "The compiler ${CMAKE_CXX_COMPILER} has no C++1y support.")
endif()

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()
#### op.cmake ends

execute_process(
  OUTPUT_VARIABLE SCANNER_CMAKE_PATH
  COMMAND python3 -c "import scannerpy.stdlib.build_flags as b; b.print_cmake()")
include(${SCANNER_CMAKE_PATH})
message("scanner cmake ${SCANNER_CMAKE_PATH}")

find_package(CUDA REQUIRED)
add_definitions(-DHAVE_CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
if(COMPILER_SUPPORTS_CXX1Y)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

build_op(
  LIB_NAME prepare_image
  CPP_SRCS prepare_image.cc
)

build_op(
  LIB_NAME extraction_op
  CPP_SRCS extraction_op.cc
  PROTO_SRC siftExtraction.proto
)

build_op(
  LIB_NAME sequential_matching
  CPP_SRCS sequential_matching.cc
  PROTO_SRC colmap.proto
)

build_op(
  LIB_NAME incremental_mapping
  CPP_SRCS incremental_mapping.cc
  PROTO_SRC incremental_mapping.proto
)

build_op(
  LIB_NAME merge_mapping
  CPP_SRCS merge_mapping.cc
  PROTO_SRC merge_mapping.proto
)

build_op(
  LIB_NAME prepare_patch_match
  CPP_SRCS prepare_patch_match.cc
  PROTO_SRC prepare_patch_match.proto
)

build_op(
  LIB_NAME patch_match
  CPP_SRCS patch_match.cc
)

include_directories(${COLMAP_INCLUDE_DIRS})
link_directories(${COLMAP_LINK_DIRS})

find_library(COLMAP_LIB colmap HINTS ${COLMAP_LINK_DIRS})
find_library(FLANN_LIB flann HINTS ${COLMAP_LINK_DIRS})
find_library(GRACLUS_LIB graclus HINTS ${COLMAP_LINK_DIRS})
find_library(LSD_LIB lsd HINTS ${COLMAP_LINK_DIRS})
find_library(PBA_LIB pba HINTS ${COLMAP_LINK_DIRS})
find_library(POISSON_RECON_LIB poisson_recon HINTS ${COLMAP_LINK_DIRS})
find_library(SQLITE3_LIB sqlite3 HINTS ${COLMAP_LINK_DIRS})
find_library(SIFT_GPU_LIB sift_gpu HINTS ${COLMAP_LINK_DIRS})
find_library(VLFEAT_LIB vlfeat HINTS ${COLMAP_LINK_DIRS})
find_library(COLMAP_CUDA_LIB colmap_cuda HINTS ${COLMAP_LINK_DIRS})

SET (
  COLMAP_STATIC_LIB
  ${COLMAP_LIB}
  ${FLANN_LIB}
  ${GRACLUS_LIB}
  ${LSD_LIB}
  ${PBA_LIB}
  ${POISSON_RECON_LIB}
  ${SQLITE3_LIB}
  ${SIFT_GPU_LIB}
  ${VLFEAT_LIB}
  ${COLMAP_CUDA_LIB}
)

SET(STATIC_WHOLE_ARCHIVE -Wl,--whole-archive ${COLMAP_STATIC_LIB} -Wl,--no-whole-archive)

# remove these from the list of libraries, as we want the linker to use
# the static libraries from above
list(REMOVE_ITEM COLMAP_LIBRARIES colmap flann graclus lsd pba poisson_recon sqlite3 sift_gpu vlfeat colmap_cuda)
SET(COLMAP_LIBRARIES ${STATIC_WHOLE_ARCHIVE} ${COLMAP_LIBRARIES})

target_link_libraries(prepare_image PUBLIC ${COLMAP_LIBRARIES})
target_link_libraries(extraction_op PUBLIC ${COLMAP_LIBRARIES})
target_link_libraries(sequential_matching PUBLIC ${COLMAP_LIBRARIES})
target_link_libraries(incremental_mapping PUBLIC ${COLMAP_LIBRARIES})
target_link_libraries(merge_mapping PUBLIC ${COLMAP_LIBRARIES})
target_link_libraries(prepare_patch_match PUBLIC ${COLMAP_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(patch_match PUBLIC ${COLMAP_LIBRARIES} ${CUDA_LIBRARIES})
