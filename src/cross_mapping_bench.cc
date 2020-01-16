#include <algorithm>
#include <iostream>

#include <argh.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

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
void find_embedding_dim(HighFive::File file, std::vector<uint32_t> &optimal_E,
                        uint32_t max_E, const DataFrame &df, bool verbose)
{
    // max_E=20, tau=1, Tp=1
    auto embedding_dim =
        std::unique_ptr<EmbeddingDim>(new T(max_E, 1, 1, verbose));

    optimal_E.resize(df.n_columns());

    for (auto i = 0u; i < df.n_columns(); i++) {
        const auto ts = df.columns[i];
        const auto best_E = embedding_dim->run(ts);

        if (verbose) {
            std::cout << "Find embedding dimension for column #" << i << " done"
                      << std::endl;
        }

        optimal_E[i] = best_E;
    }

    const auto dataspace = HighFive::DataSpace::From(optimal_E);
    auto dataset = file.createDataSet<uint32_t>("/embedding", dataspace);
    dataset.write(optimal_E);
}

template <class T>
void cross_mapping(HighFive::File file, uint32_t max_E, const DataFrame &df,
                   const std::vector<uint32_t> &optimal_E, bool verbose)
{
    std::vector<float> rhos(df.n_columns());

    // max_E=20, tau=1, Tp=0
    auto xmap = std::unique_ptr<CrossMapping>(new T(max_E, 1, 0, verbose));

    const auto dataspace =
        HighFive::DataSpace({df.n_columns(), df.n_columns()});
    auto dataset = file.createDataSet<float>("/corrcoef", dataspace);

    for (auto i = 0u; i < df.n_columns(); i++) {
        const auto library = df.columns[i];

        xmap->run(rhos, library, df.columns, optimal_E);

        if (verbose) {
            std::cout << "Cross mapping for column #" << i << " done"
                      << std::endl;
        }

        dataset.select({i, 0}, {1, df.n_columns()}).write(rhos);
    }
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

    HighFive::File file("output.h5", HighFive::File::Overwrite);
    std::vector<uint32_t> optimal_E;

    timer_simplex.start();

    if (kernel_type == "cpu") {
        std::cout << "Using CPU Simplex kernel" << std::endl;

        find_embedding_dim<EmbeddingDimCPU>(file, optimal_E, max_E, df,
                                            verbose);
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU Simplex kernel" << std::endl;

        find_embedding_dim<EmbeddingDimGPU>(file, optimal_E, max_E, df,
                                            verbose);
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

        cross_mapping<CrossMappingCPU>(file, max_E, df, optimal_E, verbose);
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU cross mapping kernel" << std::endl;

        cross_mapping<CrossMappingGPU>(file, max_E, df, optimal_E, verbose);
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
