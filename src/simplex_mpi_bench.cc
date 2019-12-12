#include <iostream>

#include "dataset.h"
#include "mpi_master.h"
#include "mpi_worker.h"
#include "simplex.h"
#include "simplex_cpu.h"
#include "timer.h"

class SimplexMPIMaster : public MPIMaster
{
public:
    SimplexMPIMaster(const std::string fname, MPI_Comm comm)
        : MPIMaster(comm), dataset(fname), current_id(0)
    {
    }
    ~SimplexMPIMaster() {}

protected:
    Dataset dataset;
    int current_id;

    void next_task(nlohmann::json &task)
    {
        task["id"] = current_id;
        current_id++;
    }

    bool task_left() const { return current_id < dataset.timeseries.size(); }

    void task_done(const nlohmann::json &result)
    {
        std::cout << "Timeseries #" << result["id"] << " best E=" << result["E"]
                  << " rho=" << result["rho"] << std::endl;
    }
};

class SimplexMPIWorker : public MPIWorker
{
public:
    SimplexMPIWorker(const std::string fname, MPI_Comm comm)
        : MPIWorker(comm), dataset(fname),
          knn(new NearestNeighborsCPU(1, true)),
          simplex(new SimplexCPU(1, 1, true))
    {
    }
    ~SimplexMPIWorker() {}

protected:
    Dataset dataset;
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex;

    void do_task(nlohmann::json &result, const nlohmann::json &task)
    {
        const auto id = task["id"];

        Timeseries ts = dataset.timeseries[id];

        // Split input into two halves
        Timeseries library(ts.data(), ts.size() / 2);
        Timeseries target(ts.data() + ts.size() / 2, ts.size() / 2);

        std::vector<float> rhos;

        LUT lut;

        for (auto E = 1; E <= 20; E++) {
            knn->compute_lut(lut, library, target, E);
            const auto rho = simplex->predict(lut, library, target, E);
            rhos.push_back(rho);
        }

        const auto it = std::max_element(rhos.begin(), rhos.end());
        const auto maxE = it - rhos.begin() + 1;
        const auto maxRho = *it;

        result["id"] = id;
        result["E"] = maxE;
        result["rho"] = maxRho;
    }
};

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    if (argc < 2) {
        std::cerr << "No input" << std::endl;
        return -1;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (!rank) {
        SimplexMPIMaster master(argv[1], MPI_COMM_WORLD);
        Timer timer;

        timer.start();
        master.run();
        timer.stop();

        std::cout << "Processed dataset in " << timer.elapsed() << " [ms]"
                  << std::endl;
    } else {
        SimplexMPIWorker worker(argv[1], MPI_COMM_WORLD);

        worker.run();
    }

    MPI_Finalize();
}
