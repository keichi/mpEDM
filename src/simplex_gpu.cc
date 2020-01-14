#include <arrayfire.h>

#include "simplex_gpu.h"

Series SimplexGPU::predict(std::vector<float> &buffer, const LUT &lut,
                           const Series &target, uint32_t E)
{
    buffer.resize(lut.n_rows());

    const af::array idx(lut.n_columns(), lut.n_rows(), lut.indices.data());
    const af::array dist(lut.n_columns(), lut.n_rows(), lut.distances.data());
    const af::array target_data(target.size(), target.data());

    const af::array tmp =
        af::moddims(target_data(idx), lut.n_columns(), lut.n_rows());
    const af::array pred = af::sum(tmp * dist);

    pred.host(buffer.data());

    return Series(buffer);
}
