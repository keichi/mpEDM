include(CheckCXXCompilerFlag)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

project(cuEDM CXX)

add_executable(knn_bench)
add_executable(simplex_bench)
add_executable(ccm_bench)

target_sources(knn_bench PRIVATE src/knn_bench.cc src/dataset.cc src/lut.cc
               src/nearest_neighbors_cpu.cc)
target_sources(simplex_bench PRIVATE src/simplex_bench.cc src/dataset.cc
               src/lut.cc src/simplex_cpu.cc src/nearest_neighbors_cpu.cc)
target_sources(ccm_bench PRIVATE src/ccm_bench.cc src/dataset.cc
               src/lut.cc src/simplex_cpu.cc src/nearest_neighbors_cpu.cc)

add_subdirectory(src/thirdparty/argh)
target_link_libraries(knn_bench PRIVATE argh)

# Enable all warnings
# TODO Check if this works across all compilers we are using
add_compile_options(-Wall -Werror -Wno-unknown-pragmas)

# Enable ArrayFire backend if available
find_package(ArrayFire)
if(ArrayFire_FOUND)
  message(STATUS "ArrayFire found, building GPU backend")

  add_definitions(-DENABLE_GPU_KERNEL)

  if(APPLE)
    # Need to force CPU backend since OpenCL backend on macOS has bugs
    # See: https://github.com/arrayfire/arrayfire/issues/2687
    target_link_libraries(knn_bench PRIVATE ArrayFire::afcpu)
  else()
    target_link_libraries(knn_bench PRIVATE ArrayFire::af)
  endif()
  target_include_directories(knn_bench PRIVATE src/thirdparty/concurrentqueue)

  target_sources(knn_bench PRIVATE src/nearest_neighbors_gpu.cc)
else()
  message(WARNING "ArrayFire was NOT found, will not build GPU backend")
endif()

# Enable MPI if available
find_package(MPI)
if(MPI_CXX_FOUND)
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  add_subdirectory(src/thirdparty/json)

  add_executable(simplex_mpi_bench)
  target_sources(simplex_mpi_bench PRIVATE src/simplex_mpi_bench.cc
                 src/mpi_master.cc src/mpi_worker.cc src/dataset.cc src/lut.cc
                 src/nearest_neighbors_cpu.cc src/simplex_cpu.cc)
  target_link_libraries(simplex_mpi_bench PRIVATE MPI::MPI_CXX)
  target_link_libraries(simplex_mpi_bench PRIVATE nlohmann_json::nlohmann_json)
endif()

# Build for native CPU architecture
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL
   "RelWithDebInfo")
  check_cxx_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH)
  if(COMPILER_SUPPORTS_MARCH)
    add_compile_options("-march=native")
  endif()
endif()

# Enable OpenMP if available
# For some reason, CMake fails to detect OpenMP on clang (macOS)
find_package(OpenMP)
if(OpenMP_FOUND)
  target_link_libraries(knn_bench PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(simplex_bench PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(simplex_mpi_bench PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(ccm_bench PRIVATE OpenMP::OpenMP_CXX)
endif()

# Tests

enable_testing()

add_subdirectory(src/thirdparty/catch2)
include(CTest)

include(src/thirdparty/catch2/contrib/Catch.cmake)

# Lookup table test
add_executable(lut_test)
target_sources(lut_test PRIVATE test/lut_test.cc src/lut.cc)
target_link_libraries(lut_test PRIVATE Catch2::Catch2)
catch_discover_tests(lut_test
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)

# k-nearest neighbors test
add_executable(knn_test)
target_sources(knn_test PRIVATE test/knn_test.cc src/lut.cc src/dataset.cc
               src/nearest_neighbors_cpu.cc)
target_link_libraries(knn_test PRIVATE Catch2::Catch2)
if(ArrayFire_FOUND)
  if(APPLE)
    target_link_libraries(knn_test PRIVATE ArrayFire::afcpu)
  else()
    target_link_libraries(knn_test PRIVATE ArrayFire::af)
  endif()

  target_sources(knn_test PRIVATE src/nearest_neighbors_gpu.cc)
endif()
catch_discover_tests(knn_test
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
