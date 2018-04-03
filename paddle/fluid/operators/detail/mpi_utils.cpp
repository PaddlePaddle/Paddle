//
// Created by tangwei12 on 2018/3/27.
//

#include <stdio.h>
#include <string.h>

#include <mpi.h>
#include "mpi_utils.h"

#define max_worker_name_length 128
#define mpi_tag = 2008

namespace paddle {
    namespace operators {
        namespace detail {
            MPIUtils::MPIUtils(const std::string &worker_name) {
                InitMPI();

                int rank = 0, size = 1;
                char my_name[max_work_group_size];
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &size);
                snprintf(my_name, max_worker_name_length, worker_name.c_str());

                std::vector<char> worker_names(size * max_worker_name_length);
                MPI_Allgather(my_name, max_worker_name_length, MPI_CHAR, &worker_names[0],
                              max_worker_name_length, MPI_CHAR, MPI_COMM_WORLD);
                for (int i = 0; i < number_of_procs; i++) {
                    name_to_id_[std::string(&worker_names[i * 128])] = i;
                }
            }

            void MPIUtils::InitMPI() {
                int flag = 0;
                MPI_CHECK(MPI_Initialized(&flag));

                if (!flag) {
                    int rank = 0, size = 1, len = -1;
                    char host_name[max_worker_name_length];

                    MPI_Init(0, 0);
                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                    MPI_Comm_size(MPI_COMM_WORLD, &size);
                    MPI_Get_processor_name(host_name, &len);
                }
            };


            MPISend::MPISend(const Meta &meta) {
                done1_ = 1;
                done2_ = 0;
                this->meta = meta;
            }

            MPISend::Send() {
                MPI_Send(&meta.request, meta.count, meta.datatype, meta.dst, meta.tag,
                         MPI_COMM_WORLD);
                done2_ = 1;
            }

            bool MPISend::IsReady() {
                return true;
            }

            bool MPISend::IsFinished() { return done1_ && done2_; }

            MPISend::~MPISend() { MPI_Free_mem(meta); }


            MPIRecv::MPIRecv(const Meta &meta) {
                this->meta = meta;
            }

            MPIRecv::Recv() {}

            bool MPIRecv::IsReady() {
                return true;
            }

            MPIRecv::IsFinished() {}

            MPIRecv::~MPIRecv() {
                MPI_Free_mem(meta);
            }

        }  // namespace detail

    }  // namespace operators
}  // namespace paddle