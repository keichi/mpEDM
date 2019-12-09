#include <iostream>

#include "mpi_common.h"
#include "mpi_master.h"

// Based on https://github.com/nepda/pi-pp/blob/master/serie_4
void MPIMaster::run()
{
    nlohmann::json task;
    MPI_Status stat;

    while (task_left() || !working.empty()) {
        // Wait for any incomming message
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &stat);

        // Store rank of receiver into worker
        const auto worker = stat.MPI_SOURCE;

        // Worker asked for a task
        if (stat.MPI_TAG == TAG_ASK_FOR_TASK) {
            MPI_Recv(nullptr, 0, MPI_BYTE, worker, TAG_ASK_FOR_TASK, comm,
                     &stat);

            if (task_left()) {
                next_task(task);

                const auto send_buf = task.dump();
                // We have unprocessed tasks, we send one task to the worker
                MPI_Send(send_buf.c_str(), send_buf.size(), MPI_BYTE, worker,
                         TAG_TASK_DATA, comm);

                // Mark worker as working on a task
                working.insert(worker);
            } else {
                // Send stop msg to worker
                MPI_Send(nullptr, 0, MPI_BYTE, worker, TAG_STOP, comm);
            }

            // Worker sent result
        } else if (stat.MPI_TAG == TAG_RESULT) {
            auto count = 0;
            MPI_Get_count(&stat, MPI_BYTE, &count);

            std::vector<uint8_t> recv_buf(count);

            // We got a result message
            MPI_Recv(recv_buf.data(), count, MPI_BYTE, worker, TAG_RESULT, comm,
                     &stat);

            task_done(nlohmann::json::parse(recv_buf));

            // Mark worker as stopped
            working.erase(worker);

            if (!task_left()) {
                // Send stop msg to worker
                MPI_Send(nullptr, 0, MPI_BYTE, worker, TAG_STOP, comm);
            }
        }
    }
}
