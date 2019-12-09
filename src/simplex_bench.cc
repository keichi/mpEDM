#include <iostream>

#include "dataset.h"
#include "simplex.h"
#include "simplex_cpu.h"

void simplex_projection(Simplex &simplex, const Timeseries &ts)
{
    // Split input into two halves
    Timeseries library(ts.data(), ts.size() / 2);
    Timeseries predictee(ts.data() + ts.size() / 2, ts.size() / 2);

    std::cout << "E\trho" << std::endl;
    for (int E = 1; E <= 20; E++) {
        float rho = simplex.predict(library, predictee, E);

        std::cout << E << "\t" << rho << std::endl;
    }
}

int main(int argc, char *argv[])
{
    Dataset ds(argv[1]);

    // tau=1, k=30, Tp=1
    SimplexCPU simplex(1, 30, 1, true);

    auto i = 0;
    for (const auto &ts : ds.timeseries) {
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Simplex projection for timeseries #" << (i++)
                  << std::endl;

        simplex_projection(simplex, ts);
    }

    return 0;
}
