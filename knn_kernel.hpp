#ifndef __KNN_KERNEL_HPP__
#define __KNN_KERNEL_HPP__

#include <iostream>

#include "dataset.hpp"
#include "lut.hpp"
#include "timer.hpp"

class KNNKernel
{
public:
    KNNKernel(int E_max, int tau, int k) : E_max(E_max), tau(tau), top_k(k) {}

    virtual ~KNNKernel() {}

    virtual void run(const Dataset &ds)
    {
        for (int i = 0; i < ds.n_cols; i++) {
            const float *const col = ds.cols[i].data();

            Timer timer;
            timer.start();

            for (int E = 1; E <= E_max; E++) {
                LUT out;
                int n = ds.n_rows - (E - 1) * tau;
                compute_lut(out, col, E, n);
            }

            timer.stop();

            std::cout << "Computed LUT for column #" << i << " in "
                      << timer.elapsed() << " [ms]" << std::endl;
        }
    }

    virtual void compute_lut(LUT &out, const float *const col, int E,
                             int n) = 0;

protected:
    // Pointer to the timeseries we are working on
    // Note that we do NOT own memory, Dataset holds it
    const float *col;
    // Maximum embedding dimension (number of columns)
    const int E_max;
    // Lag
    const int tau;
    // Number of neighbors to find
    const int top_k;
};

#endif
