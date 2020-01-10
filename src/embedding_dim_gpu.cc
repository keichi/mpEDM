#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <arrayfire.h>

#include "embedding_dim_gpu.h"
#include "stats.h"

// clang-format off
uint32_t EmbeddingDimGPU::run(const Timeseries &ts)
{
    #pragma omp parallel num_threads(n_devs)
    {
        #ifdef _OPENMP
        uint32_t dev_id = omp_get_thread_num();
        #else
        uint32_t dev_id = 0;
        #endif

        af::setDevice(dev_id);

        // Split input into two halves
        const Timeseries library(ts.data(), ts.size() / 2);
        const Timeseries target(ts.data() + ts.size() / 2, ts.size() / 2);
        Timeseries prediction;
        Timeseries shifted_target;

        #pragma omp for schedule(dynamic)
        for (auto E = 1; E <= max_E; E++) {
            knn->compute_lut(luts[dev_id], library, target, E, E + 1);
            luts[dev_id].normalize();

            simplex->predict(prediction, buffers[dev_id], luts[dev_id],
                             library, E);
            simplex->shift_target(shifted_target, target, E);

            rhos[E - 1] = corrcoef(prediction, shifted_target);
        }
    }

    const auto it = std::max_element(rhos.begin(), rhos.end());
    const auto best_E = it - rhos.begin() + 1;

    return best_E;
}
// clang-format on