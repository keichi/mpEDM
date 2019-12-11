#include <algorithm>
#include <iostream>

#include "dataset.h"
#include "simplex.h"
#include "simplex_cpu.h"
#include "timer.h"

void simplex_projection(Simplex &simplex, const Timeseries &ts)
{
    // Split input into two halves
    Timeseries library(ts.data(), ts.size() / 2);
    Timeseries predictee(ts.data() + ts.size() / 2, ts.size() / 2);

    std::vector<float> rhos;

    for (auto E = 1; E <= 20; E++) {
        const auto rho = simplex.predict(library, predictee, E);
        rhos.push_back(rho);
    }

    const auto it = std::max_element(rhos.begin(), rhos.end());
    const auto maxE = it - rhos.begin() + 1;
    const auto maxRho = *it;

    std::cout << "best E=" << maxE << " rho=" << maxRho << std::endl;
}

int main(int argc, char *argv[])
{
    const std::string fname(argv[1]);

    Timer timer_tot;

    std::cout << "Reading input dataset from " << fname << std::endl;

    timer_tot.start();

    Dataset ds(fname);

    timer_tot.stop();

    std::cout << "Read " << ds.n_rows() << " rows in " << timer_tot.elapsed()
              << " [ms]" << std::endl;

    timer_tot.start();

    // tau=1, k=30, Tp=1
    SimplexCPU simplex(1, 30, 1, true);

    auto i = 0;
    for (const auto &ts : ds.timeseries) {
        std::cout << "Simplex projection for timeseries #" << (i++) << ": ";

        simplex_projection(simplex, ts);
    }

    timer_tot.stop();

    std::cout << "Processed dataset in " << timer_tot.elapsed() << " [ms]"
              << std::endl;

    return 0;
}
