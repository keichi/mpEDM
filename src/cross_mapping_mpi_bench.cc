#include <iostream>

#include "cross_mapping_cpu.h"
#include "dataframe.h"
#include "embedding_dim_cpu.h"
#include "mpi_master.h"
#include "mpi_worker.h"
#ifdef ENABLE_GPU_KERNEL
#include "cross_mapping_gpu.h"
#include "embedding_dim_gpu.h"
#endif
#include "stats.h"
#include "timer.h"

class CrossMappingMPIMaster : public MPIMaster
{
public:
    CrossMappingMPIMaster(const std::string &fname, MPI_Comm comm)
        : MPIMaster(comm), current_id(0)
    {
        df.load(fname);
    }
    ~CrossMappingMPIMaster() {}

protected:
    DataFrame df;
    uint32_t current_id;

    void next_task(nlohmann::json &task) override
    {
        task["id"] = current_id;
        current_id++;
    }

    bool task_left() const override { return current_id < df.columns.size(); }

    void task_done(const nlohmann::json &result) override
    {
        std::cout << "Timeseries #" << result["id"] << " best E=" << result["E"]
                  << " rho=" << result["rho"] << std::endl;
    }
};

class CrossMappingMPIWorker : public MPIWorker
{
public:
    CrossMappingMPIWorker(const std::string &fname, MPI_Comm comm)
        : MPIWorker(comm), knn(new NearestNeighborsCPU(1, 1, true)),
          simplex(new SimplexCPU(1, 1, true))
    {
        df.load(fname);
    }
    ~CrossMappingMPIWorker() {}

protected:
    DataFrame df;
    std::unique_ptr<NearestNeighbors> knn;
    std::unique_ptr<Simplex> simplex;

    template <class T>
    void find_embedding_dim(std::vector<uint32_t> &optmal_E, uint32_t max_E,
                            const DataFrame &df, bool verbose)
    {
        // max_E=20, tau=1, Tp=1
        auto embedding_dim =
            std::unique_ptr<EmbeddingDim>(new T(max_E, 1, 1, verbose));

        optmal_E.resize(df.n_columns());

        for (auto i = 0u; i < df.n_columns(); i++) {
            const auto ts = df.columns[i];
            const auto best_E = embedding_dim->run(ts);

            if (verbose) {
                std::cout << "Find embedding dimension for column #" << i
                          << " done" << std::endl;
            }

            optmal_E[i] = best_E;
        }
    }

    template <class T>
    void cross_mapping(uint32_t max_E, const DataFrame &df,
                       const std::vector<uint32_t> &optimal_E, bool verbose)
    {
        std::vector<float> rhos(df.n_columns());

        // max_E=20, tau=1, Tp=0
        auto xmap = std::unique_ptr<CrossMapping>(new T(max_E, 1, 0, verbose));

        for (auto i = 0u; i < df.n_columns(); i++) {
            const auto library = df.columns[i];

            xmap->run(rhos, library, df.columns, optimal_E);

            if (verbose) {
                std::cout << "Cross mapping for column #" << i << " done"
                          << std::endl;
            }
        }
    }

    void do_task(nlohmann::json &result, const nlohmann::json &task) override
    {
        const auto id = task["id"];

        const Series ts = df.columns[id];

        // Split input into two halves
        const Series library = ts.slice(0, ts.size() / 2);
        const Series target = ts.slice(ts.size() / 2);
        Series prediction;
        Series shifted_target;

        std::vector<float> rhos(20);
        std::vector<float> buffer;

        LUT lut;

        for (auto E = 1; E <= 20; E++) {
            knn->compute_lut(lut, library, target, E);
            lut.normalize();

            const auto prediction = simplex->predict(buffer, lut, library, E);
            const auto shifted_target = simplex->shift_target(target, E);

            rhos[E - 1] = corrcoef(prediction, shifted_target);
        }

        const auto it = std::max_element(rhos.begin(), rhos.end());
        const auto max_E = it - rhos.begin() + 1;
        const auto maxRho = *it;

        result["id"] = id;
        result["E"] = max_E;
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
        CrossMappingMPIMaster master(argv[1], MPI_COMM_WORLD);
        Timer timer;

        timer.start();
        master.run();
        timer.stop();

        std::cout << "Processed dataset in " << timer.elapsed() << " [ms]"
                  << std::endl;
    } else {
        CrossMappingMPIWorker worker(argv[1], MPI_COMM_WORLD);

        worker.run();
    }

    MPI_Finalize();
}
