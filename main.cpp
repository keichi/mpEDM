#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "thirdparty/cxxopts/include/cxxopts.hpp"

#include "block.hpp"
#include "timer.hpp"

int main(int argc, char *argv[])
{
    cxxopts::Options options("cuEDM", "GPU-accelerated EDM");

    options.add_options()
        ("t,tau", "Lag",
         cxxopts::value<int>()->default_value("1"))
        ("e,emax", "Maximum embedding dimension",
         cxxopts::value<int>()->default_value("20"))
        ("k,topk", "Number of neighbors to find per point",
         cxxopts::value<int>()->default_value("100"))
        ("i,input", "Input CSV file", cxxopts::value<std::string>())
        ("h,help", "Show help");

    options.parse_positional("input");

    options.positional_help("FILE");

    auto result = options.parse(argc, argv);

    if (result["help"].as<bool>()) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    const std::string fname = result["input"].as<std::string>();
    const int tau = result["tau"].as<int>();
    const int E_max = result["emax"].as<int>();
    const int k = result["topk"].as<int>();

    Timer timer_tot;

    std::cout << "Reading input dataset from " << fname << std::endl;

    timer_tot.start();

    Dataset ds(fname);

    timer_tot.stop();

    std::cout << ds.n_rows << " rows read in " << timer_tot.elapsed() << " [ms]" << std::endl;

    timer_tot.start();

    int lut_n = ds.n_rows - (E_max - 1) * tau;
    if (lut_n <= 0) {
        std::cerr << "E or tau is too large" << std::endl;
        return 1;
    }
    if (lut_n < k) {
        std::cerr << "k is too large" << std::endl;
        return 1;
    }

    LUT cache;

    for (int i = 0; i < ds.n_cols; i++) {
        std::cout << "Computing LUT for column #" << i << std::endl;

        Timer timer;
        timer.start();

        for (int E = 1; E <= E_max; E++) {
            Block block(ds, i, E, tau);
            LUT result;

            block.compute_lut(result, k, cache);
        }

        timer.stop();

        std::cout << "LUT computed in " << timer.elapsed() << " [ms]"
                  << std::endl;
    }

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    return 0;
}
