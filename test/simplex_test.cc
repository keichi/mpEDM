#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../src/dataset.h"
#include "../src/lut.h"
#include "../src/nearest_neighbors_cpu.h"
#include "../src/simplex_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "../src/nearest_neighbors_gpu.h"
#include "../src/simplex_gpu.h"
#endif

template <class T, class U> void simplex_test_common(int E)
{
    const auto tau = 1;

    Dataset ds1, ds2;
    ds1.load("simplex_test_data.csv");
    ds2.load("simplex_test_verification_E" + std::to_string(E) + ".csv");

    Timeseries ts = ds1.timeseries[0];
    Timeseries library(ts.data(), ts.size() / 2);
    Timeseries target(ts.data() + ts.size() / 2 - (E - 1) * tau, ts.size() / 2);
    Timeseries prediction;

    auto knn = std::unique_ptr<NearestNeighbors>(new T(tau, true));
    auto simplex = std::unique_ptr<Simplex>(new U(1, 1, true));
    LUT lut;

    knn->compute_lut(lut, library, target, E, E + 1);
    lut.normalize();

    std::vector<float> buffer;

    simplex->predict(prediction, buffer, lut, library, E);

    float rmse = 0.0;

    for (size_t row = 0; row < prediction.size(); row++) {
        rmse = pow(prediction[row] - ds2.timeseries[0][row], 2);
    }

    REQUIRE(sqrt(rmse / ds2.n_rows()) < 0.0001);
}

TEST_CASE("Computed simplex is correct (CPU, E=2)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU, SimplexCPU>(2);
}

TEST_CASE("Computed simplex is correct (CPU, E=3)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU, SimplexCPU>(3);
}

TEST_CASE("Computed simplex is correct (CPU, E=4)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU, SimplexCPU>(4);
}

TEST_CASE("Computed simplex is correct (CPU, E=5)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU, SimplexCPU>(5);
}

#ifdef ENABLE_GPU_KERNEL

TEST_CASE("Computed simplex is correct (GPU, E=2)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU, SimplexGPU>(2);
}

TEST_CASE("Computed simplex is correct (GPU, E=3)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU, SimplexGPU>(3);
}

TEST_CASE("Computed simplex is correct (GPU, E=4)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU, SimplexGPU>(4);
}

TEST_CASE("Computed simplex is correct (GPU, E=5)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU, SimplexGPU>(5);
}

#endif
