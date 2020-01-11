#include <algorithm>
#include <iostream>
#include <memory>

#include <argh.h>

#include "dataset.h"
#include "embedding_dim_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "embedding_dim_gpu.h"
#endif
#include "stats.h"
#include "timer.h"

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
        "  -e, --maxe arg   Maximum embedding dimension (default: 20)\n"
        "  -p, --Tp arg     Steps to predict in future (default: 1)\n"
        "  -x, --kernel arg Kernel type {cpu|gpu|multigpu} (default: cpu)\n"
        "  -v, --verbose    Enable verbose logging (default: false)\n"
        "  -h, --help       Show help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-t", "--tau", "-p", "--tp", "-e", "--maxe", "-x",
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
    uint32_t max_E;
    cmdl({"e", "maxe"}, 20) >> max_E;
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

    std::unique_ptr<EmbeddingDim> embedding_dim;

    if (kernel_type == "cpu") {
        std::cout << "Using CPU Simplex kernel" << std::endl;

        embedding_dim = std::unique_ptr<EmbeddingDim>(
            new EmbeddingDimCPU(max_E, tau, Tp, verbose));
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU Simplex kernel" << std::endl;

        embedding_dim = std::unique_ptr<EmbeddingDim>(
            new EmbeddingDimGPU(max_E, tau, Tp, verbose));
    }
#endif
    else {
        std::cerr << "Unknown kernel type " << kernel_type << std::endl;
        return 1;
    }

    for (auto i = 0; i < ds.timeseries.size(); i++) {
        std::cout << "Simplex projection for timeseries #" << i << ": ";

        const auto best_E = embedding_dim->run(ds.timeseries[i]);

        if (verbose) {
            std::cout << "best E=" << best_E << std::endl;
        }
    }

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    return 0;
}
