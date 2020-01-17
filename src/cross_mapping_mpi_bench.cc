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

class EmbeddingDimMPIMaster : public MPIMaster
{
public:
    std::vector<uint32_t> optimal_E;

    EmbeddingDimMPIMaster(const DataFrame df, MPI_Comm comm)
        : MPIMaster(comm), current_id(0), dataframe(df)
    {
        optimal_E.resize(df.n_columns());
    }
    ~EmbeddingDimMPIMaster() {}

protected:
    uint32_t current_id;
    DataFrame dataframe;

    void next_task(nlohmann::json &task) override
    {
        task["id"] = current_id;
        current_id++;
    }

    bool task_left() const override
    {
        return current_id < dataframe.n_columns();
    }

    void task_done(const nlohmann::json &result) override
    {
        std::cout << "Timeseries #" << result["id"] << " best E=" << result["E"]
                  << std::endl;

        // result["id"] is the ID of the column
        // result["E"] is the optimal embedding dimension

        optimal_E[result["id"]] = result["E"];
    }
};

class EmbeddingDimMPIWorker : public MPIWorker
{
public:
    EmbeddingDimMPIWorker(const DataFrame df, MPI_Comm comm)
        : MPIWorker(comm), embedding_dim(new EmbeddingDimCPU(20, 1, 1, true)),
          dataframe(df)
    {
    }
    ~EmbeddingDimMPIWorker() {}

protected:
    std::unique_ptr<EmbeddingDim> embedding_dim;
    DataFrame dataframe;

    void do_task(nlohmann::json &result, const nlohmann::json &task) override
    {
        const auto id = task["id"];
        const auto ts = dataframe.columns[id];
        const auto best_E = embedding_dim->run(ts);

        if (true) {
            std::cout << "Find embedding dimension for column #" << id
                      << " done" << std::endl;
        }

        result["id"] = id;
        result["E"] = best_E;
    }
};

class CrossMappingMPIMaster : public MPIMaster
{
public:
    CrossMappingMPIMaster(const DataFrame df, MPI_Comm comm)
        : MPIMaster(comm), current_id(0), dataframe(df)
    {
    }
    ~CrossMappingMPIMaster() {}

protected:
    uint32_t current_id;
    DataFrame dataframe;

    void next_task(nlohmann::json &task) override
    {
        task["id"] = current_id;
        current_id++;
    }

    bool task_left() const override
    {
        return current_id < dataframe.n_columns();
    }

    void task_done(const nlohmann::json &result) override
    {
        std::cout << "Timeseries #" << result["id"] << " finished."
                  << std::endl;
    }
};

class CrossMappingMPIWorker : public MPIWorker
{
public:
    CrossMappingMPIWorker(const DataFrame df,
                          const std::vector<uint32_t> optimal_E, MPI_Comm comm)
        : MPIWorker(comm), xmap(new CrossMappingCPU(20, 1, 0, true)),
          dataframe(df), optimal_E(optimal_E)
    {
    }
    ~CrossMappingMPIWorker() {}

protected:
    std::unique_ptr<CrossMapping> xmap;
    DataFrame dataframe;
    std::vector<uint32_t> optimal_E;

    void do_task(nlohmann::json &result, const nlohmann::json &task) override
    {
        std::vector<float> rhos(dataframe.n_columns());

        const auto id = task["id"];
        const auto library = dataframe.columns[id];

        xmap->run(rhos, library, dataframe.columns, optimal_E);

        result["id"] = id;

        // if (true) {
        std::cout << "Cross mapping for column #" << id << " done" << std::endl;
        // }
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

    DataFrame df;
    df.load(argv[1]);

    std::vector<uint32_t> optimal_E(df.n_columns());

    Timer timer;

    if (!rank) {
        EmbeddingDimMPIMaster embedding_dim_master(df, MPI_COMM_WORLD);

        Timer timer_embedding_dim;

        timer.start();
        timer_embedding_dim.start();
        embedding_dim_master.run();
        timer_embedding_dim.stop();

        optimal_E = embedding_dim_master.optimal_E;

        std::cout << "Processed optimal E in " << timer_embedding_dim.elapsed()
                  << " [ms]" << std::endl;
    } else {
        EmbeddingDimMPIWorker embedding_dim_worker(df, MPI_COMM_WORLD);

        embedding_dim_worker.run();
    }

    MPI_Bcast(optimal_E.data(), optimal_E.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // for (size_t i = 0; i < optimal_E.size(); i++) {
    //     std::cout << i << "," << optimal_E[i] << std::endl;
    // }

    if (!rank) {
        CrossMappingMPIMaster cross_mapping_master(df, MPI_COMM_WORLD);

        Timer timer_cross_mapping;

        timer_cross_mapping.start();
        cross_mapping_master.run();
        timer_cross_mapping.stop();

        timer.stop();

        std::cout << "Processed dataset in " << timer.elapsed() << " [ms]"
                  << std::endl;
    } else {
        CrossMappingMPIWorker cross_mapping_worker(df, optimal_E,
                                                   MPI_COMM_WORLD);

        cross_mapping_worker.run();
    }

    MPI_Finalize();
}
