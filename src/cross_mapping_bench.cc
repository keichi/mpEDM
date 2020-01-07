#include <algorithm>
#include <iostream>

#include <argh.h>
#ifdef ENABLE_GPU_KERNEL
#include <arrayfire.h>
#endif

#include "cross_mapping_cpu.h"
#include "dataset.h"
#include "lut.h"
#include "nearest_neighbors_cpu.h"
#include "simplex_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "../src/nearest_neighbors_gpu.h"
#include "../src/simplex_gpu.h"
#endif
#include "stats.h"
#include "timer.h"

template <class T, class U> void simplex_projection(std::vector<uint32_t> &optmal_E, const Dataset &ds)
{
    LUT lut;

    auto knn = std::unique_ptr<NearestNeighbors>(new T(1, 1, true));
    auto simplex = std::unique_ptr<Simplex>(new U(1, 1, true));

    optmal_E.resize(ds.n_cols());

    for (auto i = 0; i < ds.n_cols(); i++) {
        const Timeseries ts = ds.timeseries[i];
        // Split input into two halves
        const Timeseries library(ts.data(), ts.size() / 2);
        const Timeseries target(ts.data() + ts.size() / 2, ts.size() / 2);
        Timeseries prediction;
        Timeseries shifted_target;

        std::vector<float> rhos;
        std::vector<float> buffer;

        for (auto E = 1; E <= 20; E++) {
            knn->compute_lut(lut, library, target, E, E + 1);
            lut.normalize();

            simplex->predict(prediction, buffer, lut, library, E);
            simplex->shift_target(shifted_target, target, E);

            const float rho = corrcoef(prediction, shifted_target);

            rhos.push_back(rho);
        }

        const auto it = std::max_element(rhos.begin(), rhos.end());
        const auto maxE = it - rhos.begin() + 1;

        std::cout << "Optimal E for column #" << i << " is " << maxE
                  << std::endl;

        optmal_E[i] = maxE;
    }
}

void cross_mapping(const Dataset &ds, const std::vector<uint32_t> &optimal_E)
{
    // E_max=20, tau=1, Tp=0, verbose=true
    CrossMappingCPU xmap(20, 1, 0, true);

    std::vector<float> rhos;

    xmap.run(rhos, ds, optimal_E);
}

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": Cross Mapping Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...] FILE\n"
        "  -t, --tau arg    Lag (default: 1)\n"
        "  -e, --emax arg   Maximum embedding dimension (default: 20)\n"
        "  -p, --Tp arg     Steps to predict in future (default: 1)\n"
        "  -x, --kernel arg Kernel type {cpu|gpu|multigpu} (default: cpu)\n"
        "  -v, --verbose    Enable verbose logging (default: false)\n"
        "  -h, --help       Show help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-t", "--tau", "-p", "--tp", "-e", "--emax", "-x",
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
    uint32_t tau;
    cmdl({"t", "tau"}, 1) >> tau;
    uint32_t Tp;
    cmdl({"p", "Tp"}, 1) >> Tp;
    uint32_t E_max;
    cmdl({"e", "emax"}, 20) >> E_max;
    std::string kernel_type;
    cmdl({"x", "kernel"}, "cpu") >> kernel_type;
    bool verbose = cmdl[{"v", "verbose"}];

    Timer timer_tot, timer_io, timer_simplex, timer_xmap;

    std::cout << "Reading input dataset from " << fname << std::endl;

    timer_tot.start();
    timer_io.start();

    Dataset ds;
    ds.load(fname);

    timer_io.stop();

    std::cout << "Read dataset (" << ds.n_rows() << " rows, " << ds.n_cols()
              << " columns) in " << timer_io.elapsed() << " [ms]" << std::endl;

    std::vector<uint32_t> optimal_E;

    timer_simplex.start();

if (kernel_type == "cpu") {
        std::cout << "Using CPU Simplex kernel" << std::endl;

        simplex_projection<NearestNeighborsCPU, SimplexCPU>(optimal_E, ds);
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU Simplex kernel" << std::endl;

        simplex_projection<NearestNeighborsGPU, SimplexGPU>(optimal_E, ds);
    }
#endif
    else {
        std::cerr << "Unknown kernel type " << kernel_type << std::endl;
        return 1;
    }

    timer_simplex.stop();

    std::cout << "Computed optimal embedding dimensions in "
              << timer_simplex.elapsed() << " [ms]" << std::endl;

    timer_xmap.start();

    cross_mapping(ds, optimal_E);

    timer_xmap.stop();

    std::cout << "Computed cross mappings in " << timer_xmap.elapsed()
              << " [ms]" << std::endl;

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    const auto xps = ds.n_cols() * ds.n_cols() * 1000 / timer_tot.elapsed();
    std::cout << xps << " cross mappings per second" << std::endl;

    return 0;
}
