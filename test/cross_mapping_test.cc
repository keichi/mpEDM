#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../src/dataset.h"
#include "../src/lut.h"
#include "../src/nearest_neighbors_cpu.h"
#include "../src/simplex_cpu.h"

// Validation data was generated with pyEDM 1.0.1:
// pyEDM.Simplex(dataFrame=pyEDM.sampleData["sardine_anchovy_sst"],
//               E=3, Tp=1, columns="anchovy", target="np_sst", lib="1 76",
//               pred="1 76", verbose=True)


TEST_CASE("Cross mapping (E=3)", "[ccm][cpu]")
{
    const uint32_t E = 3;

    Dataset ds1("sardine_anchovy_sst.csv");
    Dataset ds2("anchovy_sst_verification_E" + std::to_string(E) + ".csv");

    // tau=1, verbose=true
    auto knn =
        std::unique_ptr<NearestNeighbors>(new NearestNeighborsCPU(1, true));

    // tau=1, Tp=0, verbose=true
    auto simplex = std::unique_ptr<Simplex>(new SimplexCPU(1, 1, true));

    LUT lut;
    Timeseries library = Timeseries(ds1.timeseries[1].data(), 76);
    Timeseries target = Timeseries(ds1.timeseries[4].data(), 76);
    std::cout << "fooo" << std::endl;
    Timeseries prediction;
    Timeseries adjusted_target;
    Timeseries valid_prediction = ds2.timeseries[0];

    knn->compute_lut(lut, library, library, E);

    lut.normalize();

    simplex->predict(prediction, lut, target, E);
    simplex->adjust_target(adjusted_target, target, E);

    std::cout << prediction.size() << "\t" << valid_prediction.size() << std::endl;

    REQUIRE(prediction.size() == valid_prediction.size());

    auto mae = 0.0f;
    for (auto i = 0; i < prediction.size(); i++) {
        mae += std::abs(prediction[i] - valid_prediction[i]);
    }

    mae /= prediction.size();

    REQUIRE(mae < 0.01f);
}
