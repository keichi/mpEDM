#include <cmath>
#include <iostream>
#include <limits>

#include "simplex_cpu.h"

float min_weight = 1e-6f;

float SimplexCPU::predict(const Timeseries &library,
                          const Timeseries &predictee, int E)
{
    // Set offset to zero for forwardTau prediction
    const auto offset = (E - 1) * tau;
    const auto n_prediction = predictee.size() - offset - Tp;

    std::vector<float> prediction(n_prediction);
    LUT lut;

    knn.compute_lut(lut, library, predictee, E);

    for (auto i = 0; i < n_prediction; i++) {
        auto sum_weights = 0.0f;
        auto min_dist = std::numeric_limits<float>::max();

        for (auto j = 0; j < E + 1; j++) {
            min_dist = std::min(min_dist, lut.distance(i, j));
        }

        for (auto j = 0; j < E + 1; j++) {
            const auto idx = lut.index(i, j) + Tp;
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
    const Timeseries ts2(predictee.data() + offset + Tp, n_prediction);

    return corrcoef(ts1, ts2);
}

// Based on https://www.geeksforgeeks.org/program-find-correlation-coefficient/
float SimplexCPU::corrcoef(const Timeseries &ts1, const Timeseries &ts2)
{
    auto sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f;
    auto sum_x2 = 0.0f, sum_y2 = 0.0f;
    const auto n = std::min(ts1.size(), ts2.size());

    for (auto i = 0; i < n; i++) {
        sum_x += ts1[i];
        sum_y += ts2[i];
        sum_xy += ts1[i] * ts2[i];
        sum_x2 += ts1[i] * ts1[i];
        sum_y2 += ts2[i] * ts2[i];
    }

    return (n * sum_xy - sum_x * sum_y) /
           std::sqrt((n * sum_x2 - sum_x * sum_x) *
                     (n * sum_y2 - sum_y * sum_y));
}
