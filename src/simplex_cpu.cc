#include "simplex_cpu.h"

void SimplexCPU::predict(Series &prediction, std::vector<float> &buffer,
                         const LUT &lut, const Series &target, uint32_t E)
{
    buffer.resize(lut.n_rows());
    std::fill(buffer.begin(), buffer.end(), 0);

    for (auto i = 0u; i < lut.n_rows(); i++) {
        for (auto j = 0u; j < E + 1; j++) {
            const auto idx = lut.indices[i * lut.n_columns() + j];
            const auto dist = lut.distances[i * lut.n_columns() + j];
            buffer[i] += target[idx] * dist;
        }
    }

    prediction = Series(buffer);
}
