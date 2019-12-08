#include <iostream>

#include "nearest_neighbors.h"
#include "timer.h"

NearestNeighbors::NearestNeighbors(int E_max, int tau, int k, bool verbose)
    : E_max(E_max), tau(tau), top_k(k), verbose(verbose)
{
}

NearestNeighbors::~NearestNeighbors() {}

void NearestNeighbors::run(const Dataset &ds)
{
    auto i = 0;

    for (const auto &ts : ds.timeseries) {
        Timer timer;
        timer.start();

        for (auto E = 1; E <= E_max; E++) {
            LUT out;
            compute_lut(out, ts, E);
        }

        timer.stop();

        i++;
        if (verbose) {
            std::cout << "Computed LUT for column #" << i << " in "
                      << timer.elapsed() << " [ms]" << std::endl;
        }
    }
}
