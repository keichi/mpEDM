#ifndef __MPI_WORKER_H__
#define __MPI_WORKER_H__

#include <mpi.h>
#include <nlohmann/json.hpp>

class MPIWorker
{
public:
    MPIWorker(MPI_Comm comm) : comm(comm) {}
    virtual ~MPIWorker() {}

    void run();

protected:
    MPI_Comm comm;

    virtual void do_task(nlohmann::json &result,
                         const nlohmann::json &task) = 0;
};

#endif
