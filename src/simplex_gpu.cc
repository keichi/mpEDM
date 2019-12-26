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

    af::array af_buffer = af::constant(0, lut.n_rows());

    af::array idx(E + 1, lut.n_rows(), lut.indices.data());
    af::array dist(E + 1, lut.n_rows(), lut.distances.data());
    af::array target_data(target.size(), 1, target.data());

    af::array tmp =
        af::sum(moddims(target_data(idx + shift), E + 1, lut.n_rows()) * dist);

    tmp.host(buffer.data());

    prediction = Timeseries(buffer);
}
