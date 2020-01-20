#include <iostream>

#include <argh.h>

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

        optimal_E[result["id"]] = result["E"];
    }
};

template <class T> class EmbeddingDimMPIWorker : public MPIWorker
{
public:
    EmbeddingDimMPIWorker(const DataFrame df, bool verbose, MPI_Comm comm)
        : MPIWorker(comm),
          embedding_dim(std::unique_ptr<EmbeddingDim>(new T(20, 1, 1, true))),
          dataframe(df), verbose(verbose)
    {
    }
    ~EmbeddingDimMPIWorker() {}

protected:
    std::unique_ptr<EmbeddingDim> embedding_dim;
    DataFrame dataframe;
    bool verbose;

    void do_task(nlohmann::json &result, const nlohmann::json &task) override
    {
        const auto id = task["id"];
        const auto ts = dataframe.columns[id];
        const auto best_E = embedding_dim->run(ts);

        if (verbose) {
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

template <class T> class CrossMappingMPIWorker : public MPIWorker
{
public:
    CrossMappingMPIWorker(const DataFrame df,
                          const std::vector<uint32_t> optimal_E, bool verbose,
                          MPI_Comm comm)
        : MPIWorker(comm),
          xmap(std::unique_ptr<CrossMapping>(new T(20, 1, 0, true))),
          dataframe(df), optimal_E(optimal_E), verbose(verbose)
    {
    }
    ~CrossMappingMPIWorker() {}

protected:
    std::unique_ptr<CrossMapping> xmap;
    DataFrame dataframe;
    std::vector<uint32_t> optimal_E;
    bool verbose;

    void do_task(nlohmann::json &result, const nlohmann::json &task) override
    {
        std::vector<float> rhos(dataframe.n_columns());

        const auto id = task["id"];
        const auto library = dataframe.columns[id];

        xmap->run(rhos, library, dataframe.columns, optimal_E);

        result["id"] = id;

        if (verbose) {
            std::cout << "Cross mapping for column #" << id << " done"
                      << std::endl;
        }
    }
};

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": Cross Mapping Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...] INPUT OUTPUT\n"
        "  -t, --tau arg    Lag (default: 1)\n"
        "  -e, --maxe arg   Maximum embedding dimension (default: 20)\n"
        "  -p, --Tp arg     Steps to predict in future (default: 1)\n"
        "  -x, --kernel arg Kernel type {cpu|gpu} (default: cpu)\n"
        "  -v, --verbose    Enable verbose logging (default: false)\n"
        "  -h, --help       Show help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-t", "--tau", "-p", "--tp", "-e", "--maxe", "-x",
                       "--kernel", "-v", "--verbose"});
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

    std::string input_fname = cmdl[1];

    uint32_t tau;
    cmdl({"t", "tau"}, 1) >> tau;
    uint32_t Tp;
    cmdl({"p", "Tp"}, 1) >> Tp;
    uint32_t max_E;
    cmdl({"e", "maxe"}, 20) >> max_E;
    std::string kernel_type;
    cmdl({"x", "kernel"}, "cpu") >> kernel_type;
    bool verbose = cmdl[{"v", "verbose"}];

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
        std::cout << "Input: " << input_fname << std::endl;

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
        if (kernel_type == "cpu") {
            EmbeddingDimMPIWorker<EmbeddingDimCPU> embedding_dim_worker(
                df, verbose, MPI_COMM_WORLD);

            embedding_dim_worker.run();
        }
#ifdef ENABLE_GPU_KERNEL
        if (kernel_type == "gpu") {
            EmbeddingDimMPIWorker<EmbeddingDimGPU> embedding_dim_worker(
                df, verbose, MPI_COMM_WORLD);

            embedding_dim_worker.run();
        }
#endif
    }

    MPI_Bcast(optimal_E.data(), optimal_E.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

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
        if (kernel_type == "cpu") {
            CrossMappingMPIWorker<CrossMappingCPU> cross_mapping_worker(
                df, optimal_E, verbose, MPI_COMM_WORLD);

            cross_mapping_worker.run();
        }
#ifdef ENABLE_GPU_KERNEL
        if (kernel_type == "gpu") {
            CrossMappingMPIWorker<CrossMappingGPU> cross_mapping_worker(
                df, optimal_E, verbose, MPI_COMM_WORLD);

            cross_mapping_worker.run();
        }
#endif
    }

    MPI_Finalize();
}
