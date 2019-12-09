#include <iostream>
#include <memory>
#include <string>

#include "dataset.h"
#include "simplex_cpu.h"

int main(int argc, char *argv[])
{
    Dataset ds(argv[1]);

    SimplexCPU simplex(1, 30, 1, true);

    Timeseries ts = ds.timeseries[1];
    Timeseries library(ts.data(), ts.size() / 2);
    Timeseries predictee(ts.data() + ts.size() / 2, ts.size() / 2);

    std::cout << "E\trho" << std::endl;
    for (int E = 1; E <= 20; E++) {
        float rho = simplex.predict(library, predictee, E);

        std::cout << E  << "\t" << rho << std::endl;
    }

    return 0;
}
