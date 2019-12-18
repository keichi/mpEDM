#include <cmath>

#include "stats.h"

// clang-format off
float corrcoef(const Timeseries &x, const Timeseries &y)
{
    const auto n = std::min(x.size(), y.size());
    auto mean_x = 0.0f, mean_y = 0.0f;
    auto sum_xy = 0.0f, sum_x2 = 0.0f, sum_y2 = 0.0f;

    #pragma omp simd reduction(+:mean_x,mean_y)
    for (auto i = 0u; i < n; i++) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    #pragma omp simd reduction(+:sum_x2,sum_y2,sum_xy)
    for (auto i = 0u; i < n; i++) {
        auto diff_x = x[i] - mean_x;
        auto diff_y = y[i] - mean_y;

        sum_xy += diff_x * diff_y;
        sum_x2 += diff_x * diff_x;
        sum_y2 += diff_y * diff_y;
    }

    return sum_xy / std::sqrt(sum_x2 * sum_y2);
}
// clang-format on
