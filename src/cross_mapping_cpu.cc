#include <iostream>

#include "cross_mapping_cpu.h"
#include "stats.h"
#include "timer.h"

void CrossMappingCPU::run(std::vector<float> &rhos, const DataFrame &df,
                          const std::vector<uint32_t> &optimal_E)
{
    for (auto i = 0u; i < df.n_columns(); i++) {
        const Series library = df.columns[i];

        predict(rhos, library, df.columns, optimal_E);

        if (verbose) {
            std::cout << "Cross mapping for column #" << i << " done"
                      << std::endl;
        }
    }
}

// clang-format off
void CrossMappingCPU::predict(std::vector<float> &rhos,
                              const Series &library,
                              const std::vector<Series> &targets,
                              const std::vector<uint32_t> &optimal_E)
{
    Timer t1, t2;

    // Compute k-NN lookup tables for library timeseries
    t1.start();
    for (auto E = 1u; E <= max_E; E++) {
        knn->compute_lut(luts[E - 1], library, library, E);
        luts[E - 1].normalize();
    }
    t1.stop();

    // Compute Simplex projection from the library to every target
    t2.start();
    #pragma omp parallel
    {
        std::vector<float> buffer;

        #pragma omp for
        for (auto i = 0u; i < targets.size(); i++) {
            const auto E = optimal_E[i];

            const Series target = targets[i];
            Series prediction;
            Series shifted_target;

            simplex->predict(prediction, buffer, luts[E - 1], target, E);
            simplex->shift_target(shifted_target, target, E);

            corrcoef(prediction, shifted_target);
        }
    }
    t2.stop();

    if (verbose) {
        std::cout << "k-NN: " << t1.elapsed() << " [ms], Simplex: "
                  << t2.elapsed() << " [ms]" << std::endl;
    }
}
// clang-format on
