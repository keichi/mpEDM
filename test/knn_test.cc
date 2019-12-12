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

    Dataset ds1("knn_test_data.csv");
    Dataset ds2("knn_test_verification_E" + std::to_string(E) + ".csv");

    auto knn = std::unique_ptr<NearestNeighbors>(new T(tau, true));
    LUT lut;

    knn->compute_lut(lut, ds1.timeseries[0], ds1.timeseries[0], E, k);

    REQUIRE(lut.n_rows() == ds1.n_rows() - (E - 1));
    REQUIRE(lut.n_cols() == k);

    for (auto row = 0; row < lut.n_rows(); row++) {
        for (auto col = 0; col < lut.n_cols(); col++) {
            REQUIRE(lut.distance(row, col) == Approx(ds2.timeseries[col][row]));
        }
    }
}

TEST_CASE("Computed k-NN lookup table is correct (CPU, E=2)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsCPU>(2);
}

TEST_CASE("Computed k-NN lookup table is correct (CPU, E=3)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsCPU>(3);
}

TEST_CASE("Computed k-NN lookup table is correct (CPU, E=4)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsCPU>(4);
}

TEST_CASE("Computed k-NN lookup table is correct (CPU, E=5)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsCPU>(5);
}

#ifdef ENABLE_GPU_KERNEL

TEST_CASE("Computed k-NN lookup table is correct (GPU, E=2)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsGPU>(2);
}

TEST_CASE("Computed k-NN lookup table is correct (GPU, E=3)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsGPU>(3);
}

TEST_CASE("Computed k-NN lookup table is correct (GPU, E=4)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsGPU>(4);
}

TEST_CASE("Computed k-NN lookup table is correct (GPU, E=5)", "[knn][cpu]")
{
    knn_test_common<NearestNeighborsGPU>(5);
}

#endif
