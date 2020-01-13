#include "simplex_cpu.h"

void SimplexCPU::predict(Timeseries &prediction, std::vector<float> &buffer,
                         const LUT &lut, const Timeseries &target, uint32_t E)
{
    const auto shift = (E - 1) * tau + Tp;

    buffer.resize(lut.n_rows());
    std::fill(buffer.begin(), buffer.end(), 0);

    for (auto i = 0u; i < lut.n_rows(); i++) {
        for (auto j = 0u; j < E + 1; j++) {
            const auto idx = lut.indices[i * lut.n_cols() + j];
            const auto dist = lut.distances[i * lut.n_cols() + j];
            buffer[i] += target[idx + shift] * dist;
        }
    }

    prediction = Timeseries(buffer);
}
