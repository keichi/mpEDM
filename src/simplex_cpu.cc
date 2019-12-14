#include <cmath>
#include <iostream>
#include <limits>

#include "simplex_cpu.h"

void SimplexCPU::predict(Timeseries &prediction, const LUT &lut,
                         const Timeseries &target, int E)
{
    const auto shift = (E - 1) * tau + Tp;

    std::fill(_prediction.begin(), _prediction.end(), 0);
    _prediction.resize(lut.n_rows());

    for (auto i = 0; i < lut.n_rows(); i++) {
        for (auto j = 0; j < E + 1; j++) {
            const auto idx = lut.index(i, j);
            const auto dist = lut.distance(i, j);
            _prediction[i] += target[idx + shift] * dist;
        }
    }

    prediction = Timeseries(_prediction);
}
