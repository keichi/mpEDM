#include <algorithm>

#include "embedding_dim_cpu.h"
#include "stats.h"

uint32_t EmbeddingDimCPU::run(const Series &ts)
{
    // Split input into two halves
    const Series library = ts.slice(0, ts.size() / 2);
    const Series target = ts.slice(ts.size() / 2);
    Series prediction;
    Series shifted_target;

    for (auto E = 1u; E <= max_E; E++) {
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
