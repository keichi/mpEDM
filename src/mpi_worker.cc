#include <iostream>

#include "mpi_common.h"
#include "mpi_worker.h"

// Based on https://github.com/nepda/pi-pp/blob/master/serie_4
void MPIWorker::run()
{
    MPI_Status stat;

    while (true) {
        // Here we send a message to the master asking for a task
        MPI_Send(nullptr, 0, MPI_BYTE, 0, TAG_ASK_FOR_TASK, comm);

        // Wait for a reply from master
        MPI_Probe(0, MPI_ANY_TAG, comm, &stat);

        // We got a task
        if (stat.MPI_TAG == TAG_TASK_DATA) {
            auto count = 0;
            MPI_Get_count(&stat, MPI_BYTE, &count);

            std::vector<uint8_t> recv_buf(count);

            // Retrieve task data from master into msg_buffer
            MPI_Recv(recv_buf.data(), count, MPI_BYTE, 0, TAG_TASK_DATA, comm,
                     &stat);

            // Work on task
            nlohmann::json result;
            do_task(result, nlohmann::json::from_cbor(recv_buf));

            // Send result to master
            const auto send_buf = nlohmann::json::to_cbor(result);
            MPI_Send(send_buf.data(), send_buf.size(), MPI_BYTE, 0, TAG_RESULT,
                     comm);

            // We got a stop message
        } else if (stat.MPI_TAG == TAG_STOP) {
            MPI_Recv(nullptr, 0, MPI_BYTE, 0, TAG_STOP, comm, &stat);
            break;
        }
    }
}
