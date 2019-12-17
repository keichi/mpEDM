#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../src/dataset.h"
#include "../src/lut.h"
#include "../src/nearest_neighbors_cpu.h"
#include "../src/simplex_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "../src/nearest_neighbors_gpu.h"
#endif

template <class T> void simplex_test_common(int E)
{
    const auto tau = 1;

    Dataset ds1("simplex_test_data.csv");
    Dataset ds2("simplex_test_verification_E" + std::to_string(E) + ".csv");

    Timeseries library(ds1.timeseries[0].data(), ds1.timeseries[0].size() / 2);
    Timeseries target(ds1.timeseries[0].data() + ds1.timeseries[0].size() / 2 - (E-1)*tau, ds1.timeseries[0].size() / 2);
    Timeseries prediction;

    auto knn = std::unique_ptr<NearestNeighbors>(new T(tau, true));
    // auto simplex = std::unique_ptr<Simplex>(new T(1, 1, true));
    // NearestNeighborsCPU knn(1, true);
    SimplexCPU simplex(1, 1, true);
    LUT lut;

    knn->compute_lut(lut, library, target, E, E+1);
    lut.normalize();
    simplex.predict(prediction, lut, library, E);

    float rmse = 0.0;

    for (size_t row = 0; row < prediction.size(); row++) {
        // std::cout << prediction[row] << " | " << ds2.timeseries[0][row] << std::endl;
        rmse = pow(prediction[row] - ds2.timeseries[0][row], 2);
    }

    rmse = sqrt(rmse/ds2.n_rows());
    // std::cout << "RMSE: " << rmse << " (" << prediction.size() << ")" << std::endl;

    REQUIRE(rmse < 0.0001);
}

TEST_CASE("Computed simplex is correct (CPU, E=2)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU>(2);
}

TEST_CASE("Computed simplex is correct (CPU, E=3)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU>(3);
}

TEST_CASE("Computed simplex is correct (CPU, E=4)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU>(4);
}

TEST_CASE("Computed simplex is correct (CPU, E=5)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU>(5);
}

#ifdef ENABLE_GPU_KERNEL

TEST_CASE("Computed simplex is correct (GPU, E=2)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU>(2);
}

TEST_CASE("Computed simplex is correct (GPU, E=3)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU>(3);
}

TEST_CASE("Computed simplex is correct (GPU, E=4)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU>(4);
}

TEST_CASE("Computed simplex is correct (GPU, E=5)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU>(5);
}

#endif
