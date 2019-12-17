#include <iostream>

#include <omp.h>

#include "cross_mapping_cpu.h"
#include "stats.h"
#include "timer.h"

void CrossMappingCPU::predict(std::vector<float> &rhos,
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
    #pragma omp parallel for
    for (auto i = 0; i < targets.size(); i++) {
        const auto tid = omp_get_thread_num();
        const auto E = optimal_E[i];

        const Timeseries target = targets[i];
        Timeseries prediction;
        Timeseries shifted_target;

        simplex[tid].predict(prediction, luts[E - 1], target, E);
        simplex[tid].shift_target(shifted_target, target, E);

        corrcoef(prediction, shifted_target);
    }
    t2.stop();

    std::cout << "LUT: " << t1.elapsed() << " [ms], Simplex: "
              << t2.elapsed() << " [ms]" << std::endl;
}
