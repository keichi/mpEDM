#include <iostream>
#include <memory>

#include "dataset.h"
#include "lut.h"
#include "nearest_neighbors_cpu.h"
#include "simplex_cpu.h"
#include "stats.h"
#include "timer.h"

void cross_mapping(const Timeseries &library, const Timeseries &target)
{
    const int E = 3;

    // tau=1, verbose=true
    auto knn =
        std::unique_ptr<NearestNeighbors>(new NearestNeighborsCPU(1, true));

    // tau=1, Tp=0, verbose=true
    auto simplex = std::unique_ptr<Simplex>(new SimplexCPU(1, 1, true));

    LUT lut;
    Timeseries prediction;
    Timeseries adjusted_target;

    knn->compute_lut(lut, library, library, E);

    lut.normalize();

    simplex->predict(prediction, lut, target, E);
    simplex->adjust_target(adjusted_target, target, E);


    for (auto i = 0; i < prediction.size(); i++) {
        std::cout << adjusted_target[i] << "\t" << prediction[i] << std::endl;
    }

    const auto rho = corrcoef(prediction, adjusted_target);
    std::cout << "rho=" << rho << std::endl;
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

    std::vector<float> rhos;

    // for (auto i = 0; i < ds.timeseries.size(); i++) {
        // for (auto j = 0; j < ds.timeseries.size(); j++) {
            // if (i == j) {
                // continue;
            // }

            // std::cout << i << "->" << j << std::endl;
            // cross_mapping(ds.timeseries[i], ds.timeseries[j]);
        // }
    // }

    cross_mapping(ds.timeseries[1], ds.timeseries[4]);
    // cross_mapping(ds.timeseries[4], ds.timeseries[1]);

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    return 0;
}
