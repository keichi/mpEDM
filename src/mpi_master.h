#ifndef __MPI_MASTER_H__
#define __MPI_MASTER_H__

#include <unordered_set>

#include <mpi.h>
#include <nlohmann/json.hpp>

class MPIMaster
{
public:
    MPIMaster(MPI_Comm comm) : comm(comm) {}
    virtual ~MPIMaster() {}

    void run();

protected:
    MPI_Comm comm;
    std::unordered_set<int> working;

    virtual void next_task(nlohmann::json &task) = 0;
    virtual bool task_left() const = 0;
    virtual void task_done(const nlohmann::json &result){};
};

#endif
