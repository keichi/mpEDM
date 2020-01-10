#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <arrayfire.h>

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

    // Compute k-NN lookup tables for library timeseries
    t1.start();
    #pragma omp parallel num_threads(n_devs)
    {
        #ifdef _OPENMP
        uint32_t dev_id = omp_get_thread_num();
        #else
        uint32_t dev_id = 0;
        #endif

        af::setDevice(dev_id);

        // Compute lookup tables for library timeseries
        #pragma omp for schedule(dynamic)
        for (auto E = 1; E <= max_E; E++) {
            knn->compute_lut(luts[E - 1], library, library, E);
            luts[E - 1].normalize();
        }
    }
    t1.stop();

    // Compute Simplex projection from the library to every target
    t2.start();
    #pragma omp parallel
    {
        std::vector<float> buffer;

        #pragma omp for
        for (auto i = 0; i < targets.size(); i++) {
            const auto E = optimal_E[i];

            const Timeseries target = targets[i];
            Timeseries prediction;
            Timeseries shifted_target;

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
