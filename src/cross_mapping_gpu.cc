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
        for (auto E = 1; E <= max_E; E++) {
            knn->compute_lut(luts[E - 1], library, library, E);
            luts[E - 1].normalize();
        }
    }
    t1.stop();

    // Compute Simplex projection from the library to every target
    for (auto i = 0; i < ds.timeseries.size(); i++) {
        const auto E = optimal_E[i];

        af::array target(data.col(i));
        af::array prediction;
        af::array shifted_target;

        simplex(prediction, luts[E - 1], target, E);
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

void CrossMappingGPU::simplex(af::array &prediction, const LUT &lut,
                              const af::array &target, uint32_t E)
{
    const auto shift = (E - 1) * tau + Tp;

    const af::array idx(lut.n_cols(), lut.n_rows(), lut.indices.data());
    const af::array dist(lut.n_cols(), lut.n_rows(), lut.distances.data());

    const af::array tmp =
        af::moddims(target(idx + shift), lut.n_cols(), lut.n_rows());
    prediction = af::sum(tmp * dist);
}

void CrossMappingGPU::shift_target(af::array &shifted_target, const af::array &target,
                  uint32_t E)
{
    const auto shift = (E - 1) * tau + Tp;
    const auto n_prediction = target.dims(0);

    shifted_target =  target(af::seq(shift, n_prediction - 1));
}