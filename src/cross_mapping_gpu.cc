#include <iostream>

#include "cross_mapping_gpu.h"
#include "stats.h"
#include "timer.h"

void CrossMappingGPU::run(std::vector<float> &rhos, const Dataset &ds,
                          const std::vector<uint32_t> &optimal_E)
{
    for (auto i = 0; i < ds.n_cols(); i++) {
        const Timeseries library = ds.timeseries[i];

        predict(rhos, library, ds.timeseries, optimal_E);

        if (verbose) {
            std::cout << "Cross mapping for column #" << i << " done"
                      << std::endl;
        }
    }
}

// clang-format off
void CrossMappingGPU::predict(std::vector<float> &rhos,
                              const Timeseries &library,
                              const std::vector<Timeseries> &targets,
                              const std::vector<uint32_t> &optimal_E)
{
    Timer t1, t2;

    t1.start();
    // Compute lookup tables for library timeseries
    for (auto E = 1; E <= E_max; E++) {
        knn->compute_lut(luts[E - 1], library, library, E);
        luts[E - 1].normalize();
    }
    t1.stop();

    t2.start();
    // Compute Simplex projection from the library to every target
    std::vector<float> buffer;

    for (auto i = 0; i < targets.size(); i++) {
        const auto E = optimal_E[i];

        const Timeseries target = targets[i];
        Timeseries prediction;
        Timeseries shifted_target;

        simplex->predict(prediction, buffer, luts[E - 1], target, E);
        simplex->shift_target(shifted_target, target, E);

        corrcoef(prediction, shifted_target);
    }
    
    t2.stop();

    std::cout << "k-NN: " << t1.elapsed() << " [ms], Simplex: "
              << t2.elapsed() << " [ms]" << std::endl;
}
// clang-format on
