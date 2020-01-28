#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../src/data_frame.h"
#include "../src/lut.h"
#include "../src/nearest_neighbors_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "../src/nearest_neighbors_gpu.h"
#endif

template <class T> void knn_test_common(int E)
{
    const auto tau = 1;
    const auto Tp = 0;
    const auto k = 4;

    DataFrame df1, df2;
    df1.load_csv("knn_test_data.csv");
    df2.load_csv("knn_test_validation_E" + std::to_string(E) + ".csv");

    auto knn = std::unique_ptr<NearestNeighbors>(new T(tau, Tp, true));
    LUT lut;

    knn->compute_lut(lut, df1.columns[0], df1.columns[0], E, k);

    REQUIRE(lut.n_rows() == df1.n_rows() - (E - 1));
    REQUIRE(lut.n_columns() == k);

    for (auto row = 0u; row < lut.n_rows(); row++) {
        for (auto col = 0u; col < lut.n_columns(); col++) {
            REQUIRE(lut.distances[row * lut.n_columns() + col] ==
                    Approx(df2.columns[col][row]));
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

TEST_CASE("Compute k-NN lookup table (GPU, E=2)", "[knn][gpu]")
{
    knn_test_common<NearestNeighborsGPU>(2);
}

TEST_CASE("Compute k-NN lookup table (GPU, E=3)", "[knn][gpu]")
{
    knn_test_common<NearestNeighborsGPU>(3);
}

TEST_CASE("Compute k-NN lookup table (GPU, E=4)", "[knn][gpu]")
{
    knn_test_common<NearestNeighborsGPU>(4);
}

TEST_CASE("Compute k-NN lookup table (GPU, E=5)", "[knn][gpu]")
{
    knn_test_common<NearestNeighborsGPU>(5);
}

#endif
