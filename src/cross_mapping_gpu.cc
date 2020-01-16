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

// clang-format off
void CrossMappingGPU::run(std::vector<float> &rhos, const Series &library,
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

            const auto target = targets[i];
            const auto prediction =
                simplex->predict(buffer, luts[E - 1], target, E);
            const auto shifted_target = simplex->shift_target(target, E);

            rhos[i] = corrcoef(prediction, shifted_target);
        }
    }
    t2.stop();

    if (verbose) {
        std::cout << "k-NN: " << t1.elapsed() << " [ms], Simplex: "
                  << t2.elapsed() << " [ms]" << std::endl;
    }
}
// clang-format on
