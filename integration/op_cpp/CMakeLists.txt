cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

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

function(build_op)
  set(options)
  set(oneValueArgs LIB_NAME PROTO_SRC NO_FLAGS)
  set(multiValueArgs CPP_SRCS)
  cmake_parse_arguments(args "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include_directories("${CMAKE_CURRENT_BINARY_DIR}")

  # Build protobuf files if they exist
  if(NOT("${args_PROTO_SRC}" STREQUAL ""))
    find_package(Protobuf REQUIRED)
    set(PROTOBUF_IMPORT_DIRS "${SCANNER_PATH}")
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${args_PROTO_SRC})
    protobuf_generate_python(PROTO_PY ${args_PROTO_SRC})
    add_custom_target(${args_LIB_NAME}_proto_files
      DEPENDS ${PROTO_HDRS} ${PROTO_PY})
    add_library(${args_LIB_NAME} SHARED ${args_CPP_SRCS} ${PROTO_SRCS})
    add_dependencies(${args_LIB_NAME} ${args_LIB_NAME}_proto_files)
    target_link_libraries(${args_LIB_NAME} PUBLIC "${PROTOBUF_LIBRARY}")
  else()
    add_library(${args_LIB_NAME} SHARED ${args_CPP_SRCS})
  endif()

  # NO_FLAGS is primarily for special treatment of libstdlib right now
  if("${args_NO_FLAGS}" STREQUAL "")
    # Explictly link libscanner.so
    execute_process(
      OUTPUT_VARIABLE SCANNER_LIB_PATH
      COMMAND
      python3 -c "import scannerpy.stdlib.build_flags as b; b.print_lib()")

    if(APPLE)
      target_link_libraries(${args_LIB_NAME} PUBLIC
        "${SCANNER_LIB_PATH}/libscanner.dylib")
    else()
      target_link_libraries(${args_LIB_NAME} PUBLIC
        "${SCANNER_LIB_PATH}/libscanner.so")
    endif()

    execute_process(
      OUTPUT_VARIABLE BUILD_FLAGS
      COMMAND
      python3 -c "import scannerpy.stdlib.build_flags as b; b.print_compile_flags()")
    set_target_properties(
      ${args_LIB_NAME} PROPERTIES
      COMPILE_FLAGS "${BUILD_FLAGS}")

    execute_process(
      OUTPUT_VARIABLE LINK_FLAGS
      COMMAND
      python3 -c "import scannerpy.stdlib.build_flags as b; b.print_link_flags()")
    set_target_properties(
      ${args_LIB_NAME} PROPERTIES
      LINK_FLAGS "${LINK_FLAGS}")
  endif()
endfunction()
#### op.cmake ends

# execute_process(
#   OUTPUT_VARIABLE SCANNER_CMAKE_PATH
#   COMMAND python3 -c "import scannerpy.stdlib.build_flags as b; b.print_cmake()")
# include(${SCANNER_CMAKE_PATH})
# message("scanner cmake ${SCANNER_CMAKE_PATH}")

# build_op will create a shared library called lib${LIB_NAME}.so that builds
# from all of the CPP_SRCS. You can also optionally specify a PROTO_SRC that
# points to a Protobuf file and will generate the C++ and Python bindings.
build_op(
  LIB_NAME prepare_image
  CPP_SRCS prepare_image.cc
)

build_op(
  LIB_NAME extraction_op
  CPP_SRCS extraction_op.cc
  PROTO_SRC siftExtraction.proto)

build_op(
  LIB_NAME sequential_matching
  CPP_SRCS sequential_matching.cc
  PROTO_SRC colmap.proto
)

build_op(
  LIB_NAME incremental_mapping
  CPP_SRCS incremental_mapping.cc
)

include_directories(${COLMAP_INCLUDE_DIRS})
link_directories(${COLMAP_LINK_DIRS})

SET (MYLIB -Wl,--whole-archive ${COLMAP_LIBRARIES} -Wl,--no-whole-archive)

find_library(COLMAP_LIB colmap HINTS ${COLMAP_LINK_DIRS})
find_library(FLANN_LIB flann HINTS ${COLMAP_LINK_DIRS})
find_library(GRACLUS_LIB graclus HINTS ${COLMAP_LINK_DIRS})
find_library(LSD_LIB lsd HINTS ${COLMAP_LINK_DIRS})
find_library(PBA_LIB pba HINTS ${COLMAP_LINK_DIRS})
find_library(POISSON_RECON_LIB poisson_recon HINTS ${COLMAP_LINK_DIRS})
find_library(SQLITE3_LIB sqlite3 HINTS ${COLMAP_LINK_DIRS})
find_library(SIFT_GPU_LIB sift_gpu HINTS ${COLMAP_LINK_DIRS})
find_library(VLFEAT_LIB vlfeat HINTS ${COLMAP_LINK_DIRS})

target_link_libraries(prepare_image PUBLIC ${COLMAP_LIB} ${FLANN_LIB} ${GRACLUS_LIB} ${LSD_LIB} ${PBA_LIB} ${POISSON_RECON_LIB} ${SQLITE3_LIB} ${SIFT_GPU_LIB} ${VLFEAT_LIB})
target_link_libraries(prepare_image PUBLIC ${MYLIB})

target_link_libraries(extraction_op PUBLIC ${COLMAP_LIB} ${FLANN_LIB} ${GRACLUS_LIB} ${LSD_LIB} ${PBA_LIB} ${POISSON_RECON_LIB} ${SQLITE3_LIB} ${SIFT_GPU_LIB} ${VLFEAT_LIB})
target_link_libraries(extraction_op PUBLIC ${MYLIB})

target_link_libraries(sequential_matching PUBLIC ${COLMAP_LIB} ${FLANN_LIB} ${GRACLUS_LIB} ${LSD_LIB} ${PBA_LIB} ${POISSON_RECON_LIB} ${SQLITE3_LIB} ${SIFT_GPU_LIB} ${VLFEAT_LIB})
target_link_libraries(sequential_matching PUBLIC ${MYLIB})

target_link_libraries(incremental_mapping PUBLIC ${COLMAP_LIB} ${FLANN_LIB} ${GRACLUS_LIB} ${LSD_LIB} ${PBA_LIB} ${POISSON_RECON_LIB} ${SQLITE3_LIB} ${SIFT_GPU_LIB} ${VLFEAT_LIB})
target_link_libraries(incremental_mapping PUBLIC ${MYLIB})
