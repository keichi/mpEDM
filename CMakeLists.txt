include(CheckCXXCompilerFlag)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(mpEDM CXX)

# Enable all warnings
# TODO Check if this works across all compilers we are using
add_compile_options(-Wall -Werror -Wno-unknown-pragmas)

# Build for native CPU architecture
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL
   "RelWithDebInfo")
  check_cxx_compiler_flag(-march=native CXX_SUPPORTS_MARCH)
  if(CXX_SUPPORTS_MARCH)
    add_compile_options("-march=native")
  endif()
endif()

add_library(mpedm SHARED src/data_frame.cc src/lut.cc
            src/nearest_neighbors_cpu.cc
            src/simplex_cpu.cc src/cross_mapping_cpu.cc
            src/embedding_dim_cpu.cc src/stats.cc)

add_executable(knn_bench src/knn_bench.cc)
add_executable(simplex_bench src/simplex_bench.cc)
add_executable(cross_mapping_bench src/cross_mapping_bench.cc)

target_link_libraries(knn_bench PRIVATE mpedm)
target_link_libraries(simplex_bench PRIVATE mpedm)
target_link_libraries(cross_mapping_bench PRIVATE mpedm)

# argh
add_subdirectory(src/thirdparty/argh)
target_link_libraries(knn_bench PRIVATE argh)
target_link_libraries(simplex_bench PRIVATE argh)
target_link_libraries(cross_mapping_bench PRIVATE argh)

# HDF5
find_package(HDF5 REQUIRED)
set(USE_BOOST OFF)
set(HIGHFIVE_UNIT_TESTS OFF)
set(HIGHFIVE_EXAMPLES OFF)
add_subdirectory(src/thirdparty/highfive)
target_link_libraries(mpedm PRIVATE HighFive)
target_link_libraries(cross_mapping_bench PRIVATE HighFive)

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_FOUND)
  message(STATUS "OpenMP found, enabling multi-threading")

  target_link_libraries(mpedm PRIVATE OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP was NOT found, multi-threading is disabled")
endif()

# Enable GPU backend if ArrayFire and OpenMP are available
find_package(ArrayFire)
if(ArrayFire_FOUND AND OpenMP_FOUND)
  message(STATUS "ArrayFire found, enabling GPU backend")

  add_definitions(-DENABLE_GPU_KERNEL)

  if(APPLE)
    # Need to force the CPU backend since the OpenCL backend has bugs on macOS
    # See: https://github.com/arrayfire/arrayfire/issues/2687
    target_link_libraries(mpedm PRIVATE ArrayFire::afcpu)
  else()
    target_link_libraries(mpedm PRIVATE ArrayFire::af)
  endif()

  target_sources(mpedm PRIVATE src/nearest_neighbors_gpu.cc src/simplex_gpu.cc
                 src/cross_mapping_gpu.cc src/embedding_dim_gpu.cc)
else()
  message(WARNING "ArrayFire was NOT found, GPU backend is disabled")
endif()

# Enable MPI if available
find_package(MPI)
if(MPI_CXX_FOUND)
  message(STATUS "MPI found, enabling multi-node benchmarks")

  set(JSON_BuildTests OFF CACHE INTERNAL "")
  add_subdirectory(src/thirdparty/json)

  target_sources(mpedm PRIVATE src/mpi_master.cc src/mpi_worker.cc)
  target_link_libraries(mpedm PRIVATE MPI::MPI_CXX
                        nlohmann_json::nlohmann_json)

  add_executable(simplex_mpi_bench src/simplex_mpi_bench.cc)
  add_executable(cross_mapping_mpi_bench src/cross_mapping_mpi_bench.cc)

  target_link_libraries(simplex_mpi_bench PRIVATE mpedm MPI::MPI_CXX
                        nlohmann_json::nlohmann_json)
  target_link_libraries(cross_mapping_mpi_bench PRIVATE mpedm MPI::MPI_CXX
                        nlohmann_json::nlohmann_json)

  target_link_libraries(cross_mapping_mpi_bench PRIVATE argh)

  if(HDF5_IS_PARALLEL)
    target_link_libraries(cross_mapping_mpi_bench PRIVATE HighFive)
  else()
    message(FATAL_ERROR "HDF5 is not a parallel build")
  endif()

else()
    message(WARNING "MPI was NOT found, multi-node benchmarks are disabled")
endif()

# Tests

enable_testing()

add_subdirectory(src/thirdparty/catch2)
include(CTest)

include(src/thirdparty/catch2/contrib/Catch.cmake)

# Lookup table test
add_executable(lut_test test/lut_test.cc)
target_link_libraries(lut_test PRIVATE mpedm Catch2::Catch2)
catch_discover_tests(lut_test
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)

# k-nearest neighbors test
add_executable(knn_test test/knn_test.cc)
target_link_libraries(knn_test PRIVATE mpedm Catch2::Catch2)
catch_discover_tests(knn_test
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)

# Cross mapping test (one-to-one)
add_executable(xmap_one_to_one_test test/xmap_one_to_one_test.cc)
target_link_libraries(xmap_one_to_one_test PRIVATE mpedm Catch2::Catch2)
catch_discover_tests(xmap_one_to_one_test
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)

# Cross mapping test (all-to-all)
add_executable(xmap_all_to_all_test test/xmap_all_to_all_test.cc)
target_link_libraries(xmap_all_to_all_test PRIVATE mpedm Catch2::Catch2 HighFive)
catch_discover_tests(xmap_all_to_all_test
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)

# Simplex test
add_executable(simplex_test test/simplex_test.cc)
target_link_libraries(simplex_test PRIVATE mpedm Catch2::Catch2)
catch_discover_tests(simplex_test
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
