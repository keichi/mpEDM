#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <arrayfire.h>

#include "cross_mapping_gpu.h"
#include "stats.h"
#include "timer.h"

void CrossMappingGPU::run(std::vector<float> &rhos, const Dataset &ds,
                          const std::vector<uint32_t> &optimal_E)
{
    rhos.resize(ds.n_cols());

    af::array data(ds.n_rows(), ds.n_cols(), ds.data());

    for (auto i = 0; i < ds.n_cols(); i++) {

        predict(rhos, ds, data, i, optimal_E);

        if (verbose) {
            std::cout << "Cross mapping for column #" << i << " done"
                      << std::endl;
        }
    }
}

// clang-format off
void CrossMappingGPU::predict(std::vector<float> &rhos,
                              const Dataset &ds,
                              af::array data,
                              const uint32_t index,
                              const std::vector<uint32_t> &optimal_E)
{
    Timer t1, t2;

    const Timeseries library = ds.timeseries[index];
    
    LUT lut;
    std::vector<af::array> idx(max_E);
    std::vector<af::array> dist(max_E);

    t1.start();
    // Compute lookup tables for library timeseries
    for (auto E = 1; E <= max_E; E++) {
        knn->compute_lut(lut, library, library, E);
        lut.normalize();

        idx[E - 1] = af::array(lut.n_cols(), lut.n_rows(), lut.indices.data());
        dist[E - 1] = af::array(lut.n_cols(), lut.n_rows(), lut.distances.data());
    }
    t1.stop();

    t2.start();
    // Compute Simplex projection from the library to every target
    for (auto i = 0; i < ds.timeseries.size(); i++) {
        const auto E = optimal_E[i];

        af::array target(data.col(i));
        af::array prediction;
        af::array shifted_target;

        simplex(prediction, idx[E - 1], dist[E - 1], target, E);
        shift_target(shifted_target, target, E);

        rhos[i] = af::corrcoef<float>(prediction, af::transpose(shifted_target));
    }
    t2.stop();

    if (verbose) {
        std::cout << "k-NN: " << t1.elapsed() << " [ms], Simplex: "
                  << t2.elapsed() << " [ms]" << std::endl;
    }
}
// clang-format on

void CrossMappingGPU::simplex(af::array &prediction,
                              const af::array &idx, const af::array &dist,
                              const af::array &target, uint32_t E)
{
    const auto shift = (E - 1) * tau + Tp;

    const af::array tmp =
        af::moddims(target(idx + shift), idx.dims(0), idx.dims(1));
    prediction = af::sum(tmp * dist);
}

void CrossMappingGPU::shift_target(af::array &shifted_target,
                                   const af::array &target, uint32_t E)
{
    const auto shift = (E - 1) * tau + Tp;
    const auto n_prediction = target.dims(0);

    shifted_target = target(af::seq(shift, n_prediction - 1));
}