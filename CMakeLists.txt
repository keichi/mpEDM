include(CheckCXXCompilerFlag)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

project(cuEDM CXX)

add_executable(cuEDM_kNN edm_knn.cc)

add_subdirectory(thirdparty/argh)
target_link_libraries(cuEDM_kNN PRIVATE argh)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  check_cxx_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH)
  if (COMPILER_SUPPORTS_MARCH)
    target_compile_options(cuEDM_kNN PRIVATE "-march=native")
  endif()
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
  target_link_libraries(cuEDM_kNN PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(ArrayFire)
if (ArrayFire_FOUND)
  message(STATUS "ArrayFire found, building GPU backend")
  add_definitions(-DENABLE_GPU_KERNEL)
  target_link_libraries(cuEDM_kNN PRIVATE ArrayFire::af)
  target_include_directories(cuEDM_kNN PRIVATE thirdparty/concurrentqueue)
else()
  message(WARNING "ArrayFire was NOT found, will not build GPU backend")
endif()
