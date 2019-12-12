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
        for (auto j = 0; j < E + 1; j++) {
            const auto idx = lut.index(i, j);
            const auto dist = lut.distance(i, j);
            _prediction[i] += target[idx + offset] * dist;
        }
    }

    prediction = Timeseries(_prediction);
}
