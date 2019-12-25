#include <cmath>
#include <iostream>
#include <limits>

#include <arrayfire.h>

#include "simplex_gpu.h"

void SimplexGPU::predict(Timeseries &prediction, std::vector<float> &buffer,
                         const LUT &lut, const Timeseries &target, uint32_t E)
{
    const auto shift = (E - 1) * tau + Tp;

    std::fill(buffer.begin(), buffer.end(), 0);
    buffer.resize(lut.n_rows());

    af::array af_buffer(0, lut.n_rows());

    af_print(af_buffer);

    for (auto i = 0u; i < lut.n_rows(); i++) {
        for (auto j = 0u; j < E + 1; j++) {
            const auto idx = lut.indices[i * lut.n_cols() + j];
            const auto dist = lut.distances[i * lut.n_cols() + j];
            buffer[i] += target[idx + shift] * dist;
        }
    }

    prediction = Timeseries(buffer);
}
