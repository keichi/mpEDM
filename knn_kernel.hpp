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

    void load_column(const Dataset &ds, int col_idx)
    {
        col = ds.cols[col_idx].data();
        n_rows = ds.n_rows;
    }

    void run()
    {
        LUT out;

        for (int E = 1; E <= E_max; E++) {
            int n = n_rows - (E - 1) * tau;
            compute_lut(out, E, n);
        }
    }

    virtual void compute_lut(LUT &lut, int E, int n) = 0;

protected:
    // Pointer to the timeseries we are working on
    // Note that we do NOT own memory, Dataset holds it
    const float *col;
    // Number of rows in the input dataset
    int n_rows;
    // Maximum embedding dimension (number of columns)
    int E_max;
    // Lag
    int tau;
    // Number of neighbors to find
    int top_k;
};

#endif
