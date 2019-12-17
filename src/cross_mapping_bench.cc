#include <algorithm>
#include <iostream>

#include "cross_mapping_cpu.h"
#include "dataset.h"
#include "lut.h"
#include "nearest_neighbors_cpu.h"
#include "simplex_cpu.h"
#include "stats.h"
#include "timer.h"

void simplex_projection(std::vector<uint32_t> &optmal_E, const Dataset &ds)
{
    LUT lut;
    // tau=1, verbose=true
    NearestNeighborsCPU knn(1, true);
    // tau=1, Tp=1, verbose=true
    SimplexCPU simplex(1, 1, true);

    optmal_E.resize(ds.n_cols());

    for (auto i = 0; i < ds.n_cols(); i++) {
        const Timeseries ts = ds.timeseries[i];
        // Split input into two halves
        const Timeseries library(ts.data(), ts.size() / 2);
        const Timeseries target(ts.data() + ts.size() / 2, ts.size() / 2);
        Timeseries prediction;
        Timeseries shifted_target;

        std::vector<float> rhos;

        for (auto E = 1; E <= 20; E++) {
            knn.compute_lut(lut, library, target, E, E + 1);
            lut.normalize();

            simplex.predict(prediction, lut, library, E);
            simplex.shift_target(shifted_target, target, E);

            const float rho = corrcoef(prediction, shifted_target);

            rhos.push_back(rho);
        }

        const auto it = std::max_element(rhos.begin(), rhos.end());
        const auto maxE = it - rhos.begin() + 1;

        // std::cout << "Optimal E for column #" << i << " is " << maxE
        // << std::endl;

        optmal_E[i] = maxE;
    }
}

void cross_mapping(const Dataset &ds, const std::vector<uint32_t> &optimal_E)
{
    // E_max=20, tau=1, Tp=0, verbose=true
    CrossMappingCPU xmap(20, 1, 0, true);

    std::vector<float> rhos;

    for (auto i = 0; i < ds.n_cols(); i++) {
        const Timeseries library = ds.timeseries[i];

        xmap.predict(rhos, library, ds.timeseries, optimal_E);

        // std::cout << "Cross mapping for column #" << i << " done" <<
        // std::endl;
    }
}

int main(int argc, char *argv[])
{
    const std::string fname(argv[1]);

    Timer timer_tot, timer_io, timer_simplex, timer_xmap;

    std::cout << "Reading input dataset from " << fname << std::endl;

    timer_tot.start();
    timer_io.start();

    Dataset ds;
    ds.load(fname);

    timer_io.stop();

    std::cout << "Read " << ds.n_rows() << " rows in " << timer_io.elapsed()
              << " [ms]" << std::endl;

    std::vector<uint32_t> optimal_E;

    timer_simplex.start();

    simplex_projection(optimal_E, ds);

    timer_simplex.stop();

    std::cout << "Computed optimal embedding dimensions in "
              << timer_simplex.elapsed() << " [ms]" << std::endl;

    timer_xmap.start();

    cross_mapping(ds, optimal_E);

    timer_xmap.stop();

    std::cout << "Computed cross mappings in " << timer_xmap.elapsed()
              << " [ms]" << std::endl;

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    return 0;
}
