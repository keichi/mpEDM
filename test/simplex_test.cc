#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../src/dataset.h"
#include "../src/lut.h"
#include "../src/nearest_neighbors_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "../src/nearest_neighbors_gpu.h"
#include "../src/simplex_gpu.h"
#endif
#include "../src/simplex_cpu.h"
#include "../src/stats.h"

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

TEST_CASE("Compute simplex projection (CPU, E=2)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU, SimplexCPU>(2);
}

TEST_CASE("Compute simplex projection (CPU, E=3)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU, SimplexCPU>(3);
}

TEST_CASE("Compute simplex projection (CPU, E=4)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU, SimplexCPU>(4);
}

TEST_CASE("Compute simplex projection (CPU, E=5)", "[simplex][cpu]")
{
    simplex_test_common<NearestNeighborsCPU, SimplexCPU>(5);
}

#ifdef ENABLE_GPU_KERNEL

TEST_CASE("Compute simplex projection (GPU, E=2)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU, SimplexGPU>(2);
}

TEST_CASE("Compute simplex projection (GPU, E=3)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU, SimplexGPU>(3);
}

TEST_CASE("Compute simplex projection (GPU, E=4)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU, SimplexGPU>(4);
}

TEST_CASE("Compute simplex projection (GPU, E=5)", "[simplex][gpu]")
{
    simplex_test_common<NearestNeighborsGPU, SimplexGPU>(5);
}

#endif

// Test data is generated using pyEDM with the following parameters:
// pyEDM.EmbedDimension(dataFrame=pyEDM.sampleData["TentMap"], lib="1 100",
//                      pred="201 500", columns="TentMap", target="TentMap",
//                      maxE=20)
template <class T, class U> void embed_dim_test_common()
{
    const auto tau = 1;
    const auto Tp = 1;
    const auto maxE = 20;

    Dataset ds1, ds2;
    ds1.load("TentMap_rEDM.csv");
    ds2.load("TentMap_rEDM_validation.csv");

    auto knn = std::unique_ptr<NearestNeighbors>(new T(tau, true));
    auto simplex = std::unique_ptr<Simplex>(new U(tau, Tp, true));

    Timeseries prediction;
    Timeseries shifted_target;
    std::vector<float> buffer;
    LUT lut;
    std::vector<float> rho(maxE);
    std::vector<float> rho_valid(maxE);

    for (auto E = 1; E <= maxE; E++) {
        const Timeseries ts = ds1.timeseries[1];
        const Timeseries library(ts.data(), 100 - 1);
        const Timeseries target(ts.data() + 200 - (E - 1) * tau,
                                300 + (E - 1) * tau);

        knn->compute_lut(lut, library, target, E);
        lut.normalize();

        simplex->predict(prediction, buffer, lut, library, E);
        simplex->shift_target(shifted_target, target, E);

        rho[E - 1] = corrcoef(prediction, shifted_target);
        rho_valid[E - 1] = ds2.timeseries[1][E - 1];

        // Check correlation coefficient
        REQUIRE(rho[E - 1] == Approx(rho_valid[E - 1]));
    }

    const auto it = std::max_element(rho.begin(), rho.end());
    const auto it2 = std::max_element(rho_valid.begin(), rho_valid.end());

    // Check optimal embedding dimension
    REQUIRE(it - rho.begin() == it2 - rho_valid.begin());
}

TEST_CASE("Find optimal embedding dimension (CPU)", "[simplex][cpu]")
{
    embed_dim_test_common<NearestNeighborsCPU, SimplexCPU>();
}
