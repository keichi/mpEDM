include(CheckCXXCompilerFlag)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

project(cuEDM CXX)

add_executable(knn_bench)

target_sources(knn_bench PRIVATE src/knn_bench.cc src/dataset.cc
               src/nearest_neighbors_cpu.cc)

add_subdirectory(src/argh)
target_link_libraries(knn_bench PRIVATE argh)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  check_cxx_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH)
  if (COMPILER_SUPPORTS_MARCH)
    target_compile_options(knn_bench PRIVATE "-march=native")
  endif()
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
  target_link_libraries(knn_bench PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(ArrayFire)
if (ArrayFire_FOUND)
  message(STATUS "ArrayFire found, building GPU backend")

  add_definitions(-DENABLE_GPU_KERNEL)

  target_link_libraries(knn_bench PRIVATE ArrayFire::af)
  target_include_directories(knn_bench PRIVATE src/concurrentqueue)

  target_sources(knn_bench PRIVATE src/nearest_neighbors_gpu.cc)
else()
  message(WARNING "ArrayFire was NOT found, will not build GPU backend")
endif()
