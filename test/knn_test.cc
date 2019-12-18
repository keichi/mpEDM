#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../src/dataset.h"
#include "../src/lut.h"
#include "../src/nearest_neighbors_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "../src/nearest_neighbors_gpu.h"
#endif

template <class T> void knn_test_common(int E)
{
    const auto tau = 1;
    const auto k = 4;

    Dataset ds1, ds2;
    ds1.load("knn_test_data.csv");
    ds2.load("knn_test_verification_E" + std::to_string(E) + ".csv");

    auto knn = std::unique_ptr<NearestNeighbors>(new T(tau, true));
    LUT lut;

    knn->compute_lut(lut, ds1.timeseries[0], ds1.timeseries[0], E, k);

    REQUIRE(lut.n_rows() == ds1.n_rows() - (E - 1));
    REQUIRE(lut.n_cols() == k);

    for (auto row = 0u; row < lut.n_rows(); row++) {
        for (auto col = 0u; col < lut.n_cols(); col++) {
            REQUIRE(lut.distances[row * lut.n_cols() + col] ==
                    Approx(ds2.timeseries[col][row]));
        }
    }
}

TEST_CASE("Compute k-NN lookup table (CPU, E=2)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsCPU>(2);
}

TEST_CASE("Compute k-NN lookup table (CPU, E=3)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsCPU>(3);
}

TEST_CASE("Compute k-NN lookup table (CPU, E=4)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsCPU>(4);
}

TEST_CASE("Compute k-NN lookup table (CPU, E=5)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsCPU>(5);
}

#ifdef ENABLE_GPU_KERNEL

TEST_CASE("Compute k-NN lookup table (GPU, E=2)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsGPU>(2);
}

TEST_CASE("Compute k-NN lookup table (GPU, E=3)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsGPU>(3);
}

TEST_CASE("Compute k-NN lookup table (GPU, E=4)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsGPU>(4);
}

TEST_CASE("Compute k-NN lookup table (GPU, E=5)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsGPU>(5);
}

#endif
