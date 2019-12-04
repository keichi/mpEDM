#include <iostream>
#include <memory>
#include <string>

#include "thirdparty/argh/argh.h"

#include "knn_kernel.hpp"
#include "knn_kernel_cpu.hpp"
#ifdef ENABLE_GPU_KERNEL
#include "knn_kernel_gpu.hpp"
#endif

#include "timer.hpp"

void usage(const std::string &app_name)
{
    std::string msg =
        app_name + ": GPU-accelerated Empirical Dynamic Modeling\n"
        "\n"
        "Usage:\n"
        "  " + app_name +" [OPTION...] FILE\n"
        "  -t, --tau arg    Lag (default: 1)\n"
        "  -e, --emax arg   Maximum embedding dimension (default: 20)\n"
        "  -k, --topk arg   Number of neighbors to find (default: 100)\n"
        "  -x, --kernel arg Kernel type {cpu|gpu} (default: cpu)\n"
        "  -h, --help       Show help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl(
        {"-t", "--tau", "-e", "--emax", "-k", "--topk", "-x", "kernel"});
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
    int E_max;
    cmdl({"e", "emax"}, 20) >> E_max;
    int top_k;
    cmdl({"k", "topk"}, 100) >> top_k;
    std::string kernel_type;
    cmdl({"x", "kernel"}, "cpu") >> kernel_type;

    std::cout << "Reading input dataset from " << fname << std::endl;

    Timer timer_tot;
    timer_tot.start();

    Dataset ds(fname);

    timer_tot.stop();

    std::cout << "Read " << ds.n_rows << " rows in " << timer_tot.elapsed()
              << " [ms]" << std::endl;

    timer_tot.start();

    int lut_n = ds.n_rows - (E_max - 1) * tau;
    if (lut_n <= 0) {
        std::cerr << "E or tau is too large" << std::endl;
        return 1;
    }
    if (lut_n < top_k) {
        std::cerr << "k is too large" << std::endl;
        return 1;
    }

    std::unique_ptr<KNNKernel> kernel;

    if (kernel_type == "cpu") {
        std::cout << "Using CPU kNN kernel" << std::endl;
        kernel =
            std::unique_ptr<KNNKernel>(new KNNKernelCPU(E_max, tau, top_k));
    }
#ifdef ENABLE_GPU_KERNEL
    else if (kernel_type == "gpu") {
        std::cout << "Using GPU kNN kernel" << std::endl;
        kernel =
            std::unique_ptr<KNNKernel>(new KNNKernelGPU(E_max, tau, top_k));
    }
#endif
    else {
        std::cerr << "Unknown kernel type " << kernel_type << std::endl;
        return 1;
    }

    for (int i = 0; i < ds.n_cols; i++) {
        Timer timer;
        timer.start();

        kernel->load_column(ds, i);
        kernel->run();

        timer.stop();

        std::cout << "Computed LUT for column #" << i << " in "
                  << timer.elapsed() << " [ms]" << std::endl;
    }

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    return 0;
}
