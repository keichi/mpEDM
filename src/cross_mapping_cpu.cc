#include <iostream>

#include "cross_mapping_cpu.h"
#include "stats.h"
#include "timer.h"

// clang-format off
void CrossMappingCPU::run(std::vector<float> &rhos, const Series &library,
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

    std::vector<float> buffer;
    // Compute Simplex projection from the library to every target
    t2.start();
    #pragma omp parallel for private(buffer) schedule(dynamic)
    for (auto i = 0u; i < targets.size(); i++) {
        const auto E = optimal_E[i];

        const auto target = targets[i];
        const auto prediction =
            simplex->predict(buffer, luts[E - 1], target, E);
        const auto shifted_target = simplex->shift_target(target, E);

        rhos[i] = corrcoef(prediction, shifted_target);
    }
    t2.stop();

    if (verbose) {
        std::cout << "k-NN: " << t1.elapsed() << " [ms], Simplex: "
                  << t2.elapsed() << " [ms]" << std::endl;
    }
}
// clang-format on
