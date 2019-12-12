#include <cmath>
#include <iostream>
#include <limits>

#include "simplex_cpu.h"

float SimplexCPU::predict(const LUT &lut, const Timeseries &library,
                          const Timeseries &target, int E)
{
    const auto offset = (E - 1) * tau + Tp;
    const auto n_prediction = target.size() - offset;

    std::vector<float> prediction(n_prediction);

    for (auto i = 0; i < n_prediction; i++) {
        auto sum_weights = 0.0f;
        auto min_dist = std::numeric_limits<float>::max();

        for (auto j = 0; j < E + 1; j++) {
            min_dist = std::min(min_dist, lut.distance(i, j));
        }

        for (auto j = 0; j < E + 1; j++) {
            const auto idx = lut.index(i, j);
            const auto dist = lut.distance(i, j);
            const auto weighted_dist =
                min_dist > 0.0f ? std::exp(-dist / min_dist) : 1.0f;
            const auto weight = std::max(weighted_dist, min_weight);

            prediction[i] += library[idx + offset] * weight;
            sum_weights += weight;
        }

        prediction[i] /= sum_weights;
    }

    const Timeseries ts1(prediction);
    const Timeseries ts2(target.data() + offset, n_prediction);

    return corrcoef(ts1, ts2);
}

// Welford's algorithm
float SimplexCPU::corrcoef(const Timeseries &x, const Timeseries &y)
{
    const auto n = std::min(x.size(), y.size());
    auto avg_x = 0.0f;
    auto avg_y = 0.0f;
    auto ssd_x = 0.0f;
    auto ssd_y = 0.0f;
    auto ssd_xy = 0.0f;

    for (auto i = 0; i < n; i++) {
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
