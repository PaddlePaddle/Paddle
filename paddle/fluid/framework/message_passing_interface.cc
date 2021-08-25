/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/framework/message_passing_interface.h"

namespace paddle {
namespace framework {

void CommonMPI::Initialize(int argc, char **argv) {
  if (argc > 0) {
    CHECK_EQ(0, MPI_Init(&argc, &argv));
  }
}

void CommonMPI::Finalizer() {}

size_t CommonMPI::Rank(CommRole comm_role) {
  int rank;
  MPI_Comm_rank(comm_role, &rank);
  return rank;
}

size_t CommonMPI::Size(CommRole comm_role) {
  int size;
  MPI_Comm_size(comm_role, &size);
  return size;
}

int CommonMPI::Split(CommRole *newcomm, CommRole comm_role, int color,
                     int key) {
  return MPI_Comm_split(comm_role, color, key, newcomm);
}

void CommonMPI::Barrier(CommRole comm_role) {
  // MPI_Barrier uses busy waiting. Try to avoid.
  // MPI_Barrier(mpi_comm());

  size_t mpi_size = Size(comm_role);
  std::vector<MPI_Request> reqs(mpi_size, MPI_REQUEST_NULL);
  int dummy = 0;

  for (size_t i = 0; i < mpi_size; i++) {
    MPI_Irecv(&dummy, 1, MPI_INT, i, 0, comm_role, &reqs[i]);
  }

  for (size_t i = 0; i < mpi_size; i++) {
    MPI_Send(&dummy, 1, MPI_INT, i, 0, comm_role);
  }

  for (size_t i = 0; i < mpi_size; i++) {
    for (unsigned long x = 1;; x = std::min(x * 2, 2000UL)) {  // NOLINT
      int flag = 0;
      MPI_Test(&reqs[i], &flag, MPI_STATUSES_IGNORE);
      if (flag) {
        break;
      }
      usleep(x);
    }
  }
}

}  // namespace framework
}  // namespace paddle
