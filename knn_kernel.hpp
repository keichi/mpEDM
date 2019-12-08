#ifndef __KNN_KERNEL_HPP__
#define __KNN_KERNEL_HPP__

#include <iostream>

#include "dataset.hpp"
#include "lut.hpp"
#include "timer.hpp"

class KNNKernel
{
public:
    KNNKernel(int E_max, int tau, int k, bool verbose)
        : E_max(E_max), tau(tau), top_k(k), verbose(verbose)
    {
    }

    virtual ~KNNKernel() {}

    virtual void run(const Dataset &ds)
    {
        auto i = 0;

        for (const auto &ts : ds.timeseries) {
            Timer timer;
            timer.start();

            for (auto E = 1; E <= E_max; E++) {
                LUT out;
                compute_lut(out, ts, E);
            }

            timer.stop();

            i++;
            if (verbose) {
                std::cout << "Computed LUT for column #" << i << " in "
                          << timer.elapsed() << " [ms]" << std::endl;
            }
        }
    }

    virtual void compute_lut(LUT &out, const Timeseries &ts, int E) = 0;

protected:
    // Maximum embedding dimension (number of columns)
    const int E_max;
    // Lag
    const int tau;
    // Number of neighbors to find
    const int top_k;
    // Enable verbose logging
    const bool verbose;
};

#endif
