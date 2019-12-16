#include <cmath>

#include "dataset.h"

// Welford's algorithm
float corrcoef(const Timeseries &x, const Timeseries &y)
{
    const auto n = std::min(x.size(), y.size());
    auto avg_x = 0.0f;
    auto avg_y = 0.0f;
    auto ssd_x = 0.0f;
    auto ssd_y = 0.0f;
    auto ssd_xy = 0.0f;

    for (auto i = 0u; i < n; i++) {
        const auto xi = x[i];
        const auto yi = y[i];
        const auto avg_x_new = avg_x + (xi - avg_x) / (i + 1);
        const auto avg_y_new = avg_y + (yi - avg_y) / (i + 1);

        ssd_x += (xi - avg_x) * (xi - avg_x_new);
        ssd_y += (yi - avg_y) * (yi - avg_y_new);
        ssd_xy += i * (xi - avg_x) * (yi - avg_y) / (i + 1);

        avg_x = avg_x_new;
        avg_y = avg_y_new;
    }

    return ssd_xy / (std::sqrt(ssd_x) * std::sqrt(ssd_y));
}
