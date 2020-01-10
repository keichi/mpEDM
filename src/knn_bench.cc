#include <iostream>
#include <memory>
#include <string>
#ifdef ENABLE_GPU_KERNEL
#include <mutex>
#endif

#include <argh.h>
#ifdef ENABLE_GPU_KERNEL
#include <arrayfire.h>
#include <concurrentqueue.h>
#endif

#include "nearest_neighbors.h"
#include "nearest_neighbors_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "nearest_neighbors_gpu.h"
#endif
#include "timer.h"

template <class T>
void run_common(const Dataset &ds, uint32_t max_E, uint32_t tau, uint32_t top_k,
                bool verbose)
{
    auto kernel = std::unique_ptr<NearestNeighbors>(new T(tau, 1, verbose));

    auto i = 0;

    for (const auto &ts : ds.timeseries) {
        Timer timer;
        timer.start();

        LUT out;

        for (auto E = 1; E <= max_E; E++) {
            kernel->compute_lut(out, ts, ts, E, top_k);
        }

        timer.stop();

        i++;
        if (verbose) {
            std::cout << "Computed LUT for column #" << i << " in "
                      << timer.elapsed() << " [ms]" << std::endl;
        }
    }
}

#ifdef ENABLE_GPU_KERNEL

moodycamel::ConcurrentQueue<uint32_t> work_queue;
std::mutex mtx;

void worker(uint32_t dev, const Dataset &ds, uint32_t max_E, uint32_t tau,
            uint32_t top_k, bool verbose)
{
    auto kernel = std::unique_ptr<NearestNeighbors>(
        new NearestNeighborsCPU(tau, 1, verbose));

    af::setDevice(dev);

    auto i = 0;

    while (work_queue.try_dequeue(i)) {
        Timer timer;
        timer.start();

        LUT out;

        for (auto E = 1; E <= max_E; E++) {
            kernel->compute_lut(out, ds.timeseries[i], ds.timeseries[i], E,
                                top_k);
        }

        timer.stop();

        if (verbose) {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "Computed LUT for column #" << i << " in "
                      << timer.elapsed() << " [ms]" << std::endl;
        }
    }
}

void run_multi_gpu(const Dataset &ds, uint32_t max_E, uint32_t tau,
                   uint32_t top_k, bool verbose)
{
    for (auto i = 0; i < ds.timeseries.size(); i++) {
        work_queue.enqueue(i);
    }

    std::vector<std::thread> threads;

    auto dev_count = af::getDeviceCount();

    for (auto dev = 0; dev < dev_count; dev++) {
        threads.push_back(
            std::thread(worker, dev, ds, max_E, tau, top_k, verbose));
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

#endif

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": k-Nearest Neighbors Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...] FILE\n"
        "  -t, --tau arg    Lag (default: 1)\n"
        "  -e, --maxe arg   Maximum embedding dimension (default: 20)\n"
        "  -k, --topk arg   Number of neighbors to find (default: 100)\n"
        "  -x, --kernel arg Kernel type {cpu|gpu|multigpu} (default: cpu)\n"
        "  -v, --verbose    Enable verbose logging (default: false)\n"
        "  -h, --help       Show help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-t", "--tau", "-e", "--maxe", "-k", "--topk", "-x",
                       "--kernel", "-v", "--verbose"});
    cmdl.parse(argc, argv);

    if (cmdl[{"-h", "--help"}]) {
        usage(cmdl[0]);
        return 0;
    }

    if (!cmdl(1)) {
        std::cerr << "No input file" << std::endl;
        usage(cmdl[0]);
        return 1;
    }

    std::string fname = cmdl[1];
    int tau;
    cmdl({"t", "tau"}, 1) >> tau;
    int max_E;
    cmdl({"e", "maxe"}, 20) >> max_E;
    int top_k;
    cmdl({"k", "topk"}, 100) >> top_k;
    std::string kernel_type;
    cmdl({"x", "kernel"}, "cpu") >> kernel_type;
    bool verbose = cmdl[{"v", "verbose"}];

    std::cout << "Reading input dataset from " << fname << std::endl;

    Timer timer_tot;
    timer_tot.start();

    Dataset ds;
    ds.load(fname);

    timer_tot.stop();

    std::cout << "Read " << ds.n_rows() << " rows in " << timer_tot.elapsed()
              << " [ms]" << std::endl;

    timer_tot.start();

    int n = ds.n_rows() - (max_E - 1) * tau;
    if (n <= 0) {
        std::cerr << "E or tau is too large" << std::endl;
        return 1;
    }
    if (n < top_k) {
        std::cerr << "k is too large" << std::endl;
        return 1;
    }

    if (kernel_type == "cpu") {
        std::cout << "Using CPU kNN kernel" << std::endl;

        run_common<NearestNeighborsCPU>(ds, max_E, tau, top_k, verbose);
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU kNN kernel" << std::endl;

        run_common<NearestNeighborsGPU>(ds, max_E, tau, top_k, verbose);
    } else if (kernel_type == "multigpu") {
        std::cout << "Using Multi-GPU kNN kernel" << std::endl;

        run_multi_gpu(ds, max_E, tau, top_k, verbose);
    }
#endif
    else {
        std::cerr << "Unknown kernel type " << kernel_type << std::endl;
        return 1;
    }

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    return 0;
}
