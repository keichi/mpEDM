#include <iostream>

#include "cross_mapping_cpu.h"
#include "stats.h"
#include "timer.h"

#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

// clang-format off
void CrossMappingCPU::run(std::vector<float> &rhos, const Series &library,
                              const std::vector<Series> &targets,
                              const std::vector<uint32_t> &optimal_E)
{
    LIKWID_MARKER_INIT;
#pragma omp parallel
    {
        LIKWID_MARKER_THREADINIT;

        LIKWID_MARKER_REGISTER("lookup");
    }


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
    #pragma omp parallel
    {
        LIKWID_MARKER_START("lookup");

        #pragma omp for private(buffer) schedule(dynamic)
        for (auto i = 0u; i < targets.size(); i++) {
            const auto E = optimal_E[i];

            const auto target = targets[i];
            const auto prediction =
                simplex->predict(buffer, luts[E - 1], target, E);
            const auto shifted_target = simplex->shift_target(target, E);

            rhos[i] = corrcoef(prediction, shifted_target);
        }

        LIKWID_MARKER_STOP("lookup");
    }
    t2.stop();

    if (verbose) {
        std::cout << "k-NN: " << t1.elapsed() << " [ms], Simplex: "
                  << t2.elapsed() << " [ms]" << std::endl;
    }

    LIKWID_MARKER_CLOSE;
}
// clang-format on
