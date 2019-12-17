#include "cross_mapping_cpu.h"
#include "stats.h"

void CrossMappingCPU::predict(const Timeseries &library,
                              const std::vector<Timeseries> &targets,
                              const std::vector<uint32_t> &optimal_E)
{
    // Compute lookup tables for library timeseries
    for (auto E = 0; E < E_max; E++) {
        knn->compute_lut(luts[E], library, library, E);
        luts[E].normalize();
    }

    // Simplex projection from the library to every target
    for (auto i = 0; i < targets.size(); i++) {
        const Timeseries target = targets[i];
        Timeseries prediction;
        Timeseries adjusted_target;
        const uint32_t E = optimal_E[i];

        simplex->predict(prediction, luts[E], target, E);
        simplex->shift_target(adjusted_target, target, E);

        corrcoef(prediction, adjusted_target);
    }
}
