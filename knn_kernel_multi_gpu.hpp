#ifndef __KNN_KERNEL_MULTI_GPU_HPP__
#define __KNN_KERNEL_MULTI_GPU_HPP__

#include <mutex>
#include <thread>
#include <vector>

#include <arrayfire.h>

#include "thirdparty/concurrentqueue/concurrentqueue.h"

#include "dataset.hpp"
#include "knn_kernel.hpp"
#include "lut.hpp"
#include "timer.hpp"

class KNNKernelMultiGPU : public KNNKernelGPU
{
public:
    KNNKernelMultiGPU(int E_max, int tau, int k) : KNNKernelGPU(E_max, tau, k)
    {
    }

    void run(const Dataset &ds)
    {
        af::info();

        n_rows = ds.n_rows;

        for (int i = 0; i < ds.n_cols; i++) {
            work_queue.enqueue(i);
        }

        std::vector<std::thread> threads;

        int dev_count = af::getDeviceCount();

        for (int dev = 0; dev < dev_count; dev++) {
            threads.push_back(std::thread(&KNNKernelMultiGPU::run_thread, this,
                                          ds, dev));
        }

        for (int i = 0; i < threads.size(); i++) {
            threads[i].join();
        }
    }

protected:
    moodycamel::ConcurrentQueue<int> work_queue;
    std::mutex mtx;

    void run_thread(const Dataset &ds, int dev)
    {

        af::setDevice(dev);

        int i;

        while (work_queue.try_dequeue(i)) {
            Timer timer;
            timer.start();

            for (int E = 1; E <= E_max; E++) {
                LUT out;
                int n = ds.n_rows - (E - 1) * tau;
                compute_lut(out, ds.cols[i].data(), E, n);
            }

            timer.stop();

            {
                std::lock_guard<std::mutex> lock(mtx);
                std::cout << "Computed LUT for column #" << i << " in "
                          << timer.elapsed() << " [ms]" << std::endl;
            }
        }
    }
};

#endif
