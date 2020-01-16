#include <algorithm>

#include "embedding_dim_cpu.h"
#include "stats.h"

uint32_t EmbeddingDimCPU::run(const Series &ts)
{
    // Split input into two halves
    const auto library = ts.slice(0, ts.size() / 2);
    const auto target = ts.slice(ts.size() / 2);

    for (auto E = 1u; E <= max_E; E++) {
        knn->compute_lut(lut, library, target, E, E + 1);
        lut.normalize();

        const auto prediction = simplex->predict(buffer, lut, library, E);
        const auto shifted_target = simplex->shift_target(target, E);

        rhos[E - 1] = corrcoef(prediction, shifted_target);
    }

    const auto it = std::max_element(rhos.begin(), rhos.end());
    const auto best_E = it - rhos.begin() + 1;

    return best_E;
}
