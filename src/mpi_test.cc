#include <chrono>
#include <iostream>
#include <thread>

#include "mpi_master.h"
#include "mpi_worker.h"

class TestMPIMaster : public MPIMaster
{
public:
    TestMPIMaster(MPI_Comm comm) : MPIMaster(comm) {}
    ~TestMPIMaster() {}

protected:
    int n_tasks = 1000;

    void next_task(nlohmann::json &task)
    {
        task["id"] = n_tasks;
        task["name"] = "test";

        std::cout << "Generating task " << task << std::endl;

        n_tasks--;
    }

    bool task_left() const { return n_tasks > 0; }

    void task_done(const nlohmann::json &result)
    {
        std::cout << "Task done: " << result << std::endl;
    }
};

class TestMPIWorker : public MPIWorker
{
public:
    TestMPIWorker(MPI_Comm comm) : MPIWorker(comm) {}
    ~TestMPIWorker() {}

protected:
    void do_task(nlohmann::json &result, const nlohmann::json &task)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        result["id"] = task["id"];
        result["status"] = "ok";
    }
};

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (!rank) {
        TestMPIMaster master(MPI_COMM_WORLD);

        master.run();
    } else {
        TestMPIWorker worker(MPI_COMM_WORLD);

        worker.run();
    }

    MPI_Finalize();
}
