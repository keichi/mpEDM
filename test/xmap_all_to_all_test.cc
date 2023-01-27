#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "../src/cross_mapping_cpu.h"
#include "../src/data_frame.h"
#include "../src/embedding_dim_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "../src/cross_mapping_gpu.h"
#include "../src/embedding_dim_gpu.h"
#endif

void xmap_all_to_all_test_common(const HighFive::File &file,
                                 std::unique_ptr<EmbeddingDim> edim,
                                 std::unique_ptr<CrossMapping> xmap)
{
    DataFrame df;
    df.load_hdf5("xmap_all_to_all_test.h5", "values");

    const auto ds_corrcoef = file.getDataSet("corrcoef");
    const auto ds_edim = file.getDataSet("embedding");

    std::vector<uint32_t> optimal_E(df.n_columns());
    std::vector<uint32_t> optimal_E_valid(df.n_columns());

    ds_edim.read(optimal_E_valid);

    for (auto i = 0u; i < df.n_columns(); i++) {
        optimal_E[i] = edim->run(df.columns[i]);

        REQUIRE(optimal_E[i] == optimal_E_valid[i]);
    }

    std::vector<float> rhos(df.n_columns());
    std::vector<float> rhos_valid(df.n_columns());

    for (auto i = 0u; i < df.n_columns(); i++) {
        xmap->run(rhos, df.columns[i], df.columns, optimal_E);
        ds_corrcoef.select({i, 0}, {1, df.n_columns()}).read(rhos_valid);

        for (auto j = 0u; j < df.n_columns(); j++) {
            REQUIRE(rhos[j] == Catch::Approx(rhos_valid[j]).margin(1e-5));
        }
    }
}

TEST_CASE("Compute all-to-all cross mappings (CPU)", "[ccm][cpu]")
{
    const HighFive::File file("xmap_all_to_all_test_validation.h5");

    auto edim =
        std::unique_ptr<EmbeddingDim>(new EmbeddingDimCPU(20, 1, 1, true));

    auto xmap =
        std::unique_ptr<CrossMapping>(new CrossMappingCPU(20, 1, 0, true));

    xmap_all_to_all_test_common(file, std::move(edim), std::move(xmap));
}

#ifdef ENABLE_GPU_KERNEL

TEST_CASE("Compute all-to-all cross mappings (GPU)", "[ccm][gpu]")
{
    const HighFive::File file("xmap_all_to_all_test_validation.h5");

    auto edim =
        std::unique_ptr<EmbeddingDim>(new EmbeddingDimGPU(20, 1, 1, true));

    auto xmap =
        std::unique_ptr<CrossMapping>(new CrossMappingGPU(20, 1, 0, true));

    xmap_all_to_all_test_common(file, std::move(edim), std::move(xmap));
}

#endif
