#ifndef __KNN_KERNEL_GPU_HPP__
#define __KNN_KERNEL_GPU_HPP__

#include "dataset.hpp"
#include "knn_kernel.hpp"
#include "lut.hpp"
#include "timer.hpp"

class KNNKernelGPU : public KNNKernel
{
public:
    KNNKernelGPU(int E_max, int tau, int k) : KNNKernel(E_max, tau, k)
    {
    }

    void compute_lut(LUT &out, int E, int n)
    {
    }

protected:
};

#endif
