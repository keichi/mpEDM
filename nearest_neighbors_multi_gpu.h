#ifndef __NEAREST_NEIGHBORS_MULTI_GPU_HPP__
#define __NEAREST_NEIGHBORS_MULTI_GPU_HPP__

#include <mutex>
#include <vector>

#include <concurrentqueue.h>

#include "dataset.h"
#include "lut.h"
#include "nearest_neighbors_gpu.h"

class NearestNeighborsMultiGPU : public NearestNeighborsGPU
{
public:
    NearestNeighborsMultiGPU(int E_max, int tau, int k, bool verbose);

    void run(const Dataset &ds);

protected:
    moodycamel::ConcurrentQueue<int> work_queue;
    std::mutex mtx;

    void run_thread(const Dataset &ds, int dev);
};

#endif
