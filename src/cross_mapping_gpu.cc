#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <arrayfire.h>

#include "cross_mapping_gpu.h"
#include "stats.h"
#include "timer.h"

CrossMappingGPU::CrossMappingGPU(uint32_t max_E, uint32_t tau, uint32_t Tp,
                                 bool verbose)
    : CrossMapping(max_E, tau, Tp, verbose),
      knn(new NearestNeighborsGPU(tau, Tp, verbose)),
      simplex(new SimplexCPU(tau, Tp, verbose)), luts(max_E)
{
    n_devs = af::getDeviceCount();
}

void CrossMappingGPU::run(std::vector<float> &rhos, const DataFrame &df,
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
void CrossMappingGPU::predict(std::vector<float> &rhos,
                              const Series &library,
                              const std::vector<Series> &targets,
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
        for (auto E = 1u; E <= max_E; E++) {
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
