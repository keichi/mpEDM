#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "timer.hpp"
#include "block.hpp"

int main(int argc, char *argv[])
{
    Dataset ds;
    ds.load_csv(argv[1]);

    std::cout << ds.n_rows << " rows read from " << argv[1] << std::endl;

    const int tau = 1;
    const int E_max = 20;

    Timer timer;

    for (int i = 0; i < ds.n_cols; i++) {
        std::cout << "Processing column #" << i << std::endl;

        timer.start();

        for (int E = 2; E <= E_max; E++) {
            Block block(ds, i, E, tau);
            std::vector<std::vector<std::pair<float, int>>> distances;

            block.calc_distances(distances);
        }

        timer.stop();

        std::cout << "Column #" << i << " processed in " << timer.elapsed()
                  << " [ms]" << std::endl;

        timer.reset();
    }

    return 0;
}
