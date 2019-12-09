#include <cmath>
#include <iostream>
#include <limits>

#include "simplex_cpu.h"

float min_weight = 1e-6f;

float SimplexCPU::predict(const Timeseries &library,
                          const Timeseries &predictee, int E)
{
    // Set delta_row to zero for forwardTau
    const auto delta_row = (E - 1) * tau;
    const auto n_prediction = predictee.size() - delta_row - Tp;

    std::vector<float> prediction(n_prediction);
    LUT lut;

    knn.compute_lut(lut, library, predictee, E);

    for (auto i = 0; i < n_prediction; i++) {
        prediction[i] = 0.0f;
        auto sum_weights = 0.0f;
        auto min_dist = std::numeric_limits<float>::max();

        for (auto j = 0; j < E + 1; j++) {
            min_dist = std::min(min_dist, lut.distance(i, j));
        }

        for (auto j = 0; j < E + 1; j++) {
            const auto idx = lut.index(i, j) + Tp;
            auto weight = 1.0f;

            if (min_dist > 0.0f) {
                weight = std::max(std::exp(-lut.distance(i, j) / min_dist),
                                  min_weight);
            }

            prediction[i] += library[idx + delta_row] * weight;
            sum_weights += weight;
        }

        prediction[i] /= sum_weights;
    }

    return corrcoef(
        Timeseries(prediction),
        Timeseries(predictee.data() + delta_row + Tp, n_prediction));
}

float SimplexCPU::corrcoef(const Timeseries &ts1, const Timeseries &ts2)
{
    auto sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f;
    auto sum_x2 = 0.0f, sum_y2 = 0.0f;

    for (auto i = 0; i < ts1.size(); i++) {
        sum_x += ts1[i];
        sum_y += ts2[i];
        sum_xy += ts1[i] * ts2[i];
        sum_x2 += ts1[i] * ts1[i];
        sum_y2 += ts2[i] * ts2[i];
    }

    return (ts1.size() * sum_xy - sum_x * sum_y) /
           std::sqrt((ts1.size() * sum_x2 - sum_x * sum_x) *
                     (ts1.size() * sum_y2 - sum_y * sum_y));
}
