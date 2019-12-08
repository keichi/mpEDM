#ifndef __KNN_KERNEL_MULTI_GPU_HPP__
#define __KNN_KERNEL_MULTI_GPU_HPP__

#include <mutex>
#include <thread>
#include <vector>

#include <arrayfire.h>
#include <concurrentqueue.h>

#include "dataset.hpp"
#include "lut.hpp"
#include "nearest_neighbors.hpp"
#include "timer.hpp"

class NearestNeighborsMultiGPU : public NearestNeighborsGPU
{
public:
    NearestNeighborsMultiGPU(int E_max, int tau, int k, bool verbose)
        : NearestNeighborsGPU(E_max, tau, k, verbose)
    {
    }

    void run(const Dataset &ds)
    {
        for (auto i = 0; i < ds.timeseries.size(); i++) {
            work_queue.enqueue(i);
        }

        std::vector<std::thread> threads;

        auto dev_count = af::getDeviceCount();

        for (auto dev = 0; dev < dev_count; dev++) {
            threads.push_back(std::thread(&NearestNeighborsMultiGPU::run_thread,
                                          this, ds, dev));
        }

        for (auto &thread : threads) {
            thread.join();
        }
    }

protected:
    moodycamel::ConcurrentQueue<int> work_queue;
    std::mutex mtx;

    void run_thread(const Dataset &ds, int dev)
    {

        af::setDevice(dev);

        auto i = 0;

        while (work_queue.try_dequeue(i)) {
            Timer timer;
            timer.start();

            for (auto E = 1; E <= E_max; E++) {
                LUT out;
                compute_lut(out, ds.timeseries[i], E);
            }

            timer.stop();

            if (verbose) {
                std::lock_guard<std::mutex> lock(mtx);
                std::cout << "Computed LUT for column #" << i << " in "
                          << timer.elapsed() << " [ms]" << std::endl;
            }
        }
    }
};

#endif
