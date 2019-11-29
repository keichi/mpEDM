#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "block.hpp"
#include "timer.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " file" << std::endl;
        return -1;
    }

    Dataset ds;
    ds.load_csv(argv[1]);

    std::cout << ds.n_rows << " rows read from " << argv[1] << std::endl;

    const int tau = 1;
    const int E_max = 20;
    const int k = 100;

    Timer timer;

    LUT cache;

    for (int i = 0; i < ds.n_cols; i++) {
        std::cout << "Computing LUT for column #" << i << std::endl;

        timer.start();

        for (int E = 1; E <= E_max; E++) {
            Block block(ds, i, E, tau);
            LUT result;

            block.compute_lut(result, k, cache);
        }

        timer.stop();

        std::cout << "LUT computed in " << timer.elapsed() << " [ms]"
                  << std::endl;

        timer.reset();
    }

    return 0;
}
