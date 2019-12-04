include(CheckCXXCompilerFlag)

cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 11)

project(cuEDM CXX)

add_executable(cuEDM main.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  check_cxx_compiler_flag(-march=native COMPILER_SUPPORTS_MARCH)
  if (COMPILER_SUPPORTS_MARCH)
    target_compile_options(cuEDM PRIVATE "-march=native")
  endif()
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
  target_compile_options(cuEDM PRIVATE ${OpenMP_CXX_FLAGS})
endif()

find_package(ArrayFire)
if (ArrayFire_FOUND)
  add_definitions(-DENABLE_GPU_KERNEL)
  target_link_libraries(cuEDM PRIVATE ArrayFire::af)
endif()
