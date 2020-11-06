#include <iostream>
#include <memory>
#include <random>
#include <string>
#ifdef ENABLE_GPU_KERNEL
#include <mutex>
#endif

#include <argh.h>
#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

#include "nearest_neighbors.h"
#include "nearest_neighbors_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "nearest_neighbors_gpu.h"
#endif
#include "timer.h"

template <class T>
void run_common(uint32_t L, uint32_t E, uint32_t tau, uint32_t iterations,
                bool verbose)
{
    LIKWID_MARKER_INIT;
#pragma omp parallel
    {
        LIKWID_MARKER_THREADINIT;

        LIKWID_MARKER_REGISTER("calc_distances");
        LIKWID_MARKER_REGISTER("partial_sort");
    }

    auto kernel = std::unique_ptr<NearestNeighbors>(new T(tau, 1, verbose));

    std::vector<float> library_vec(L);
    std::vector<float> target_vec(L);

    std::random_device rand_dev;
    std::default_random_engine engine(rand_dev());
    std::uniform_real_distribution<> dist(0.0f, 1.0f);

    for (auto i = 0u; i < L; i++) {
        library_vec[i] = dist(engine);
        target_vec[i] = dist(engine);
    }

    LUT out;
    Series library(library_vec);
    Series target(target_vec);

    // Warm-up loop
    for (auto i = 0u; i < iterations / 10; i++) {
        kernel->compute_lut(out, library, target, E, E + 1);
    }

    kernel->timer_distances.reset();
    kernel->timer_sorting.reset();

    Timer timer;
    timer.start();

    // Actual measurement loop
    for (auto i = 0u; i < iterations; i++) {
        kernel->compute_lut(out, library, target, E, E + 1);
    }

    timer.stop();

    std::cout << "Elapsed: " << timer.elapsed() << " [ms]" << std::endl;

    std::cout << "calc_distances "
              << kernel->timer_distances.elapsed() / iterations << std::endl;
    std::cout << "partial_sort " << kernel->timer_sorting.elapsed() / iterations
              << std::endl;

    LIKWID_MARKER_CLOSE;
}

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": k-Nearest Neighbors Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...]\n"
        "  -l, --length arg        Length of time series (default: 10,000)\n"
        "  -e, --embedding-dim arg Embedding dimension (default: 20)\n"
        "  -t, --tau arg           Time delay (default: 1)\n"
        "  -i, --iteration arg     Number of iterations (default: 10)\n"
        "  -x, --kernel arg        Kernel type {cpu|gpu} (default: cpu)\n"
        "  -v, --verbose           Enable verbose logging (default: false)\n"
        "  -h, --help              Show this help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-e", "--embedding-dim", "-l", "--length", "-t", "--tau",
                       "-i", "--iteration", "-x", "--kernel", "-v",
                       "--verbose"});
    cmdl.parse(argc, argv);

    if (cmdl[{"-h", "--help"}]) {
        usage(cmdl[0]);
        return 0;
    }

    int L;
    cmdl({"l", "length"}, 10000) >> L;
    int E;
    cmdl({"e", "embedding-dim"}, 20) >> E;
    int tau;
    cmdl({"t", "tau"}, 1) >> tau;
    int iterations;
    cmdl({"i", "iteration"}, 10) >> iterations;
    std::string kernel_type;
    cmdl({"x", "kernel"}, "cpu") >> kernel_type;
    bool verbose = cmdl[{"v", "verbose"}];

    if (L - (E - 1) * tau <= 0) {
        std::cerr << "E or tau is too large" << std::endl;
        return 1;
    }

    if (kernel_type == "cpu") {
        std::cout << "Using CPU kNN kernel" << std::endl;

        run_common<NearestNeighborsCPU>(L, E, tau, iterations, verbose);
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU kNN kernel" << std::endl;

        run_common<NearestNeighborsGPU>(L, E, tau, iterations, verbose);
    }
#endif
    else {
        std::cerr << "Unknown kernel type " << kernel_type << std::endl;
        return 1;
    }

    return 0;
}
