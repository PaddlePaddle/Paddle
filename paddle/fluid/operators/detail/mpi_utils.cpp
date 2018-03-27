//
// Created by tangwei12 on 2018/3/27.
//

#include <stdio.h>

#include "paddle/fluid/operators/detail/mpi_utils.h"

#define max_worker_name_length 128

namespace paddle {
namespace operators {
namespace detail {
MPIUtils::MPIUtils(const std::string& worker_name) {
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
    MPI_Get_processor_name(host_name, &len)
  }
};
}  // namespace detail

}  // namespace operators
}  // namespace paddle