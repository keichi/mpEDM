#include <cmath>
#include <iostream>
#include <limits>

#include "simplex_cpu.h"

void SimplexCPU::predict(Timeseries &prediction, const LUT &lut,
                         const Timeseries &target, int E)
{
    const auto offset = (E - 1) * tau + Tp;
    const auto n_prediction = target.size() - offset;

    std::fill(_prediction.begin(), _prediction.end(), 0);
    _prediction.resize(n_prediction);

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

            _prediction[i] += target[idx + offset] * weight;
            sum_weights += weight;
        }

        _prediction[i] /= sum_weights;
    }

    prediction = Timeseries(_prediction);
}
