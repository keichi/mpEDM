#include <algorithm>
#include <iostream>

#include <argh.h>
#include <memory>
#ifdef ENABLE_GPU_KERNEL
#include <arrayfire.h>
#endif

#include "dataset.h"
#include "nearest_neighbors.h"
#include "nearest_neighbors_cpu.h"
#include "simplex.h"
#include "simplex_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "nearest_neighbors_gpu.h"
#include "simplex_gpu.h"
#endif
#include "stats.h"
#include "timer.h"

void simplex_projection(std::shared_ptr<NearestNeighbors> knn, 
                        std::shared_ptr<Simplex> simplex,
                        const Timeseries &ts)
{
    LUT lut;

    // Split input into two halves
    Timeseries library(ts.data(), ts.size() / 2);
    Timeseries target(ts.data() + ts.size() / 2, ts.size() / 2);
    // Use following to get the exact same predictions as cppEDM
    // Timeseries target(ts.data() + ts.size() / 2 - (E - 1) * tau,
    //                   ts.size() / 2 + (E - 1) * tau);
    Timeseries prediction;
    Timeseries shifted_target;

    std::vector<float> rhos;
    std::vector<float> buffer;

    for (auto E = 1; E <= 20; E++) {
        knn->compute_lut(lut, library, target, E);
        lut.normalize();

        simplex->predict(prediction, buffer, lut, library, E);
        simplex->shift_target(shifted_target, target, E);

        const float rho = corrcoef(prediction, shifted_target);

        rhos.push_back(rho);
    }

    const auto it = std::max_element(rhos.begin(), rhos.end());
    const auto maxE = it - rhos.begin() + 1;
    const auto maxRho = *it;

    std::cout << "best E=" << maxE << " rho=" << maxRho << std::endl;
}

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": Simplex Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...] FILE\n"
        "  -t, --tau arg    Lag (default: 1)\n"
        // "  -e, --emax arg   Maximum embedding dimension (default: 20)\n"
        // "  -k, --topk arg   Number of neighbors to find (default: 100)\n"
        "  -x, --kernel arg Kernel type {cpu|gpu|multigpu} (default: cpu)\n"
        "  -v, --verbose    Enable verbose logging (default: false)\n"
        "  -h, --help       Show help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-t", "--tau", "-e", "--emax", "-k", "--topk", "-x",
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

    const std::string fname(argv[1]);
    int tau;
    cmdl({"t", "tau"}, 1) >> tau;
    // int E_max;
    // cmdl({"e", "emax"}, 20) >> E_max;
    // int top_k;
    // cmdl({"k", "topk"}, 100) >> top_k;
    std::string kernel_type;
    cmdl({"x", "kernel"}, "cpu") >> kernel_type;
    bool verbose = cmdl[{"v", "verbose"}];

    Timer timer_tot;

    std::cout << "Reading input dataset from " << fname << std::endl;

    timer_tot.start();

    Dataset ds;
    ds.load(fname);

    timer_tot.stop();

    std::cout << "Read " << ds.n_rows() << " rows in " << timer_tot.elapsed()
              << " [ms]" << std::endl;

    timer_tot.start();

    std::shared_ptr<NearestNeighbors> knn;
    std::shared_ptr<Simplex> simplex;

    if (kernel_type == "cpu") {
        std::cout << "Using CPU Simplex kernel" << std::endl;

        knn = std::shared_ptr<NearestNeighbors>(new NearestNeighborsCPU(tau, verbose));
        simplex = std::shared_ptr<Simplex>(new SimplexCPU(tau, 1, verbose));
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU Simplex kernel" << std::endl;

        knn = std::shared_ptr<NearestNeighbors>(new NearestNeighborsGPU(tau, verbose));
        simplex = std::shared_ptr<Simplex>(new SimplexGPU(tau, 1, verbose));
        
    // } else if (kernel_type == "multigpu") {
    //     std::cout << "Using Multi-GPU kNN kernel" << std::endl;

    //     run_multi_gpu(ds, E_max, tau, top_k, verbose);
    }
#endif
    else {
        std::cerr << "Unknown kernel type " << kernel_type << std::endl;
        return 1;
    }

    auto i = 0;
    for (const auto &ts : ds.timeseries) {
        std::cout << "Simplex projection for timeseries #" << (i++) << ": ";

        simplex_projection(knn, simplex, ts);
    }

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    return 0;
}
