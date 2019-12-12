#include <algorithm>
#include <iostream>

#include "dataset.h"
#include "nearest_neighbors.h"
#include "nearest_neighbors_cpu.h"
#include "simplex.h"
#include "simplex_cpu.h"
#include "stats.h"
#include "timer.h"

void simplex_projection(NearestNeighbors &knn, Simplex &simplex,
                        const Timeseries &ts)
{
    LUT lut;

    // Split input into two halves
    Timeseries library(ts.data(), ts.size() / 2);
    Timeseries target(ts.data() + ts.size() / 2, ts.size() / 2);
    Timeseries prediction;
    Timeseries adjusted_target;

    std::vector<float> rhos;

    for (auto E = 1; E <= 20; E++) {
        knn.compute_lut(lut, library, target, E);
        simplex.predict(prediction, lut, library, E);
        simplex.adjust_target(adjusted_target, target, E);

        const float rho = corrcoef(prediction, adjusted_target);

        rhos.push_back(rho);
    }

    const auto it = std::max_element(rhos.begin(), rhos.end());
    const auto maxE = it - rhos.begin() + 1;
    const auto maxRho = *it;

    std::cout << "best E=" << maxE << " rho=" << maxRho << std::endl;
}

int main(int argc, char *argv[])
{
    const std::string fname(argv[1]);

    Timer timer_tot;

    std::cout << "Reading input dataset from " << fname << std::endl;

    timer_tot.start();

    Dataset ds(fname);

    timer_tot.stop();

    std::cout << "Read " << ds.n_rows() << " rows in " << timer_tot.elapsed()
              << " [ms]" << std::endl;

    timer_tot.start();

    // tau=1, verbose=true
    NearestNeighborsCPU knn(1, true);

    // tau=1, Tp=1, verbose=true
    SimplexCPU simplex(1, 1, true);

    auto i = 0;
    for (const auto &ts : ds.timeseries) {
        std::cout << "Simplex projection for timeseries #" << (i++) << ": ";

        simplex_projection(knn, simplex, ts);
    }

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    return 0;
}
