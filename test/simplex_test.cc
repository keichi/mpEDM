#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../src/dataframe.h"
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
    const auto Tp = 1;

    DataFrame df1, df2;
    df1.load_csv("simplex_test_data.csv");
    df2.load_csv("simplex_test_validation_E" + std::to_string(E) + ".csv");

    const auto ts = df1.columns[0];
    const auto library = ts.slice(0, ts.size() / 2);
    const auto target =
        ts.slice(ts.size() / 2 - (E - 1) * tau, ts.size() - (E - 1) * tau);
    const auto valid_prediction = df2.columns[0];

    auto knn = std::unique_ptr<NearestNeighbors>(new T(tau, Tp, true));
    auto simplex = std::unique_ptr<Simplex>(new U(tau, Tp, true));
    LUT lut;

    knn->compute_lut(lut, library, target, E, E + 1);
    lut.normalize();

    std::vector<float> buffer;

    const auto prediction = simplex->predict(buffer, lut, library, E);

    REQUIRE(prediction.size() == valid_prediction.size());

    for (size_t i = 0; i < prediction.size(); i++) {
        REQUIRE(prediction[i] == Approx(valid_prediction[i]).margin(1e-4f));
    }
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
    const auto max_E = 20;

    DataFrame df1, df2;
    df1.load_csv("TentMap_rEDM.csv");
    df2.load_csv("TentMap_rEDM_validation.csv");

    auto knn = std::unique_ptr<NearestNeighbors>(new T(tau, Tp, true));
    auto simplex = std::unique_ptr<Simplex>(new U(tau, Tp, true));

    std::vector<float> buffer;
    LUT lut;
    std::vector<float> rho(max_E);
    std::vector<float> rho_valid(max_E);

    for (auto E = 1; E <= max_E; E++) {
        const auto ts = df1.columns[1];
        const auto library = ts.slice(0, 100);
        const auto target = ts.slice(200 - (E - 1) * tau, 500);

        knn->compute_lut(lut, library, target, E);
        lut.normalize();

        const auto prediction = simplex->predict(buffer, lut, library, E);
        const auto shifted_target = simplex->shift_target(target, E);

        rho[E - 1] = corrcoef(prediction, shifted_target);
        rho_valid[E - 1] = df2.columns[1][E - 1];

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

#ifdef ENABLE_GPU_KERNEL

TEST_CASE("Find optimal embedding dimension (GPU)", "[simplex][gpu]")
{
    embed_dim_test_common<NearestNeighborsGPU, SimplexGPU>();
}

#endif
