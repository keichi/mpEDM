#include <iostream>

#include "mpi_common.h"
#include "mpi_master.h"

// Based on https://github.com/nepda/pi-pp/blob/master/serie_4
// clang-format off
void MPIMaster::run()
{
    nlohmann::json task;
    MPI_Status stat;
    int comm_size;

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (comm_size < 2) {
        std::cout << "Need 2 or more processes" << std::endl;
        MPI_Abort(comm, -1);
    }

    for (auto worker = 1; worker < comm_size; worker++) {
        workers.insert(worker);
    }

    #pragma omp parallel
    {
        #pragma omp master
        while (task_left() || !workers.empty()) {
            // Wait for any incomming message
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &stat);

            const auto worker = stat.MPI_SOURCE;

            auto count = 0;
            MPI_Get_count(&stat, MPI_BYTE, &count);

            std::vector<uint8_t> recv_buf(count);

            MPI_Recv(recv_buf.data(), recv_buf.size(), MPI_BYTE, worker,
                     stat.MPI_TAG, comm, &stat);

            // Worker asked for a task
            if (stat.MPI_TAG == TAG_ASK_FOR_TASK) {
                if (!task_left()) {
                    workers.erase(worker);
                    continue;
                }

                next_task(task);
                const auto send_buf = nlohmann::json::to_cbor(task);

                MPI_Send(send_buf.data(), send_buf.size(), MPI_BYTE, worker,
                         TAG_TASK_DATA, comm);
            }
            // Worker sent result
            else if (stat.MPI_TAG == TAG_RESULT) {
                #pragma omp task firstprivate(recv_buf)
                task_done(nlohmann::json::from_cbor(recv_buf));
            }
        }
    }

    for (auto worker = 1; worker < comm_size; worker++) {
        // Send stop msg to worker
        MPI_Send(nullptr, 0, MPI_BYTE, worker, TAG_STOP, comm);
    }
}
// clang-format on
