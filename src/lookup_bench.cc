#include <algorithm>
#include <iostream>
#include <random>

#include <argh.h>
#include <omp.h>
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

#include "data_frame.h"
#include "nearest_neighbors_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "nearest_neighbors_gpu.h"
#endif
#include "simplex_cpu.h"
#include "stats.h"
#include "timer.h"

void usage(const std::string &app_name)
{
    const std::string msg =
        app_name +
        ": Lookup Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...] INPUT OUTPUT\n"
        "  -n, --num-ts arg         Number of time series (default: 10,000)\n"
        "  -l, --length arg         Length of time series (default: 10,000)\n"
        "  -e, --embedding-dim arg  Embedding dimension (default: 20)\n"
        "  -t, --tau arg            Lag (default: 1)\n"
        "  -i, --iteration arg      Number of iterations (default: 10)\n"
        "  -x, --kernel arg         Kernel type {cpu|gpu} (default: cpu)\n"
        "  -v, --verbose            Enable verbose logging (default: false)\n"
        "  -h, --help               Show help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-n", "--num-ts", "-l", "--length", "-e",
                       "--embedding-dim", "-t", "--tau", "-i", "--iteration",
                       "-x", "--kernel", "-v", "--verbose"});
    cmdl.parse(argc, argv);

    if (cmdl[{"-h", "--help"}]) {
        usage(cmdl[0]);
        return 0;
    }

    int N;
    cmdl({"n", "num-ts"}, 10000) >> N;
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

    uninitialized_vector<float> input(L * N);
    uninitialized_vector<float> output(L * N);

#pragma omp parallel
    {
        std::random_device dev;
        std::default_random_engine engine;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

#pragma omp for
        for (auto i = 0; i < N; i++) {
            for (auto j = 0; j < L; j++) {
                input[i * L + j] = dist(engine);
            }
        }
    }

    LUT lut;
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex =
        std::unique_ptr<Simplex>(new SimplexCPU(tau, 1, verbose));

    if (kernel_type == "cpu") {
        std::cout << "Using CPU Simplex kernel" << std::endl;

        knn = std::unique_ptr<NearestNeighbors>(
            new NearestNeighborsCPU(tau, 1, verbose));
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU Simplex kernel" << std::endl;

        knn = std::unique_ptr<NearestNeighbors>(
            new NearestNeighborsGPU(tau, 1, verbose));
    }
#endif
    else {
        std::cerr << "Unknown kernel type " << kernel_type << std::endl;
        return 1;
    }

    const auto library = Series(input.data(), L);

    // Compute k-NN lookup tables for library timeseries
    knn->compute_lut(lut, library, library, E);
    lut.normalize();

    LIKWID_MARKER_INIT;
#pragma omp parallel
    {
        LIKWID_MARKER_THREADINIT;

        LIKWID_MARKER_REGISTER("lookup");
    }

    Timer t;

    // Compute Simplex projection from the library to every target
#pragma omp parallel
    {
        LIKWID_MARKER_START("lookup");
    }

    for (auto iter = 0u; iter < iterations; iter++) {
        t.start();
#pragma omp parallel for
        for (auto i = 0u; i < N; i++) {
            for (auto j = 0u; j < lut.n_rows(); j++) {
                float pred = 0.0f;

                for (auto e = 0u; e < E + 1; e++) {
                    const auto idx = lut.indices[j * lut.n_columns() + e];
                    const auto dist = lut.distances[j * lut.n_columns() + e];
                    pred += input[i * L + idx] * dist;
                }

                output[i * L + j] = pred;
            }
        }
        t.stop();
    }

#pragma omp parallel
    {
        LIKWID_MARKER_STOP("lookup");
    }

    LIKWID_MARKER_CLOSE;

    std::cout << "lookup: " << t.elapsed() / iterations << std::endl;

    return 0;
}
