#include <algorithm>

#include "embedding_dim_cpu.h"
#include "stats.h"

uint32_t EmbeddingDimCPU::run(const Timeseries &ts)
{
    // Split input into two halves
    const Timeseries library(ts.data(), ts.size() / 2);
    const Timeseries target(ts.data() + ts.size() / 2, ts.size() / 2);
    Timeseries prediction;
    Timeseries shifted_target;

    for (auto E = 1; E <= max_E; E++) {
        knn->compute_lut(lut, library, target, E, E + 1);
        lut.normalize();

        simplex->predict(prediction, buffer, lut, library, E);
        simplex->shift_target(shifted_target, target, E);

        rhos[E - 1] = corrcoef(prediction, shifted_target);
    }

    const auto it = std::max_element(rhos.begin(), rhos.end());
    const auto best_E = it - rhos.begin() + 1;

    return best_E;
}
