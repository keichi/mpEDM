#include <algorithm>
#include <iostream>

#include <argh.h>

#include "cross_mapping_cpu.h"
#include "dataframe.h"
#include "embedding_dim_cpu.h"
#ifdef ENABLE_GPU_KERNEL
#include "cross_mapping_gpu.h"
#include "embedding_dim_gpu.h"
#endif
#include "stats.h"
#include "timer.h"

template <class T>
void find_embedding_dim(std::vector<uint32_t> &optmal_E, uint32_t max_E,
                        const DataFrame &df, bool verbose)
{
    // max_E=20, tau=1, Tp=1
    auto embedding_dim =
        std::unique_ptr<EmbeddingDim>(new T(max_E, 1, 1, verbose));

    optmal_E.resize(df.n_columns());

    for (auto i = 0u; i < df.n_columns(); i++) {
        const Series ts = df.columns[i];
        const auto best_E = embedding_dim->run(ts);

        if (verbose) {
            std::cout << "Optimal E for column #" << i << " is " << best_E
                      << std::endl;
        }

        optmal_E[i] = best_E;
    }
}

template <class T>
void cross_mapping(uint32_t max_E, const DataFrame &df,
                   const std::vector<uint32_t> &optimal_E, bool verbose)
{
    // max_E=20, tau=1, Tp=0
    auto xmap = std::unique_ptr<CrossMapping>(new T(max_E, 1, 0, verbose));

    std::vector<float> rhos;

    xmap->run(rhos, df, optimal_E);
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
        "  -e, --maxe arg   Maximum embedding dimension (default: 20)\n"
        "  -p, --Tp arg     Steps to predict in future (default: 1)\n"
        "  -x, --kernel arg Kernel type {cpu|gpu} (default: cpu)\n"
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

    Timer timer_tot, timer_io, timer_simplex, timer_xmap;

    std::cout << "Reading input dataset from " << fname << std::endl;

    timer_tot.start();
    timer_io.start();

    DataFrame df;
    df.load(fname);

    timer_io.stop();

    std::cout << "Read dataset (" << df.n_rows() << " rows, " << df.n_columns()
              << " columns) in " << timer_io.elapsed() << " [ms]" << std::endl;

    std::vector<uint32_t> optimal_E;

    timer_simplex.start();

    if (kernel_type == "cpu") {
        std::cout << "Using CPU Simplex kernel" << std::endl;

        find_embedding_dim<EmbeddingDimCPU>(optimal_E, max_E, df, verbose);
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU Simplex kernel" << std::endl;

        find_embedding_dim<EmbeddingDimGPU>(optimal_E, max_E, df, verbose);
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

    if (kernel_type == "cpu") {
        std::cout << "Using CPU cross mapping kernel" << std::endl;

        cross_mapping<CrossMappingCPU>(max_E, df, optimal_E, verbose);
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU cross mapping kernel" << std::endl;

        cross_mapping<CrossMappingGPU>(max_E, df, optimal_E, verbose);
    }
#endif
    else {
        std::cerr << "Unknown kernel type " << kernel_type << std::endl;
        return 1;
    }

    timer_xmap.stop();

    std::cout << "Computed cross mappings in " << timer_xmap.elapsed()
              << " [ms]" << std::endl;

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    const auto xps =
        df.n_columns() * df.n_columns() * 1000 / timer_tot.elapsed();
    std::cout << xps << " cross mappings per second" << std::endl;

    return 0;
}
