#ifndef __KNN_KERNEL_GPU_HPP__
#define __KNN_KERNEL_GPU_HPP__

#include <arrayfire.h>

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
        out.resize(n, top_k);

        af::array idx;
        af::array dist;
        
        std::vector<float> embedded_array(E * n);

        // Timer timerE;
        // timerE.start();

        // Compute distances between all points
        for (int i = 0; i < E; i++) {
            for (int j = 0; j < n; j++) {
                embedded_array[i * n + j] = col[i * tau + j];
            }
        }

        af::array embedded_data(n, E, embedded_array.data());

        // af_print(embedded_data);

        af::nearestNeighbour(idx, dist, embedded_data, embedded_data, 1, top_k, AF_SSD);
        dist = af::sqrt(dist);

        dist.host(out.distances.data());
        idx.host(out.indices.data());

        // timerE.stop();

        // std::cout << "E=" << E << " computed in " << timerE.elapsed() << " [ms]" << std::endl;

        // af_print(idx);
        // af_print(dist);
    }

protected:
};

#endif
