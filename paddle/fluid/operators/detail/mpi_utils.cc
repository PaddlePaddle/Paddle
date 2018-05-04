/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detail/mpi_utils.h"

namespace paddle {
namespace operators {
namespace detail {
void MPIIsendProcess(const void *buf, int length, int dst, int tag,
                     MPI_Request *request, MPI_Status *status) {
  MPI_Isend(buf, length, MPI_BYTE, dst, tag, MPI_COMM_WORLD, request);
  MPI_Wait(request, status);
}

void MPIIrecvProcess(void *buf, int length, int src, int tag,
                     MPI_Request *request, MPI_Status *status) {
  MPI_Irecv(buf, length, MPI_BYTE, src, tag, MPI_COMM_WORLD, request);
  MPI_Wait(request, status);
}
}  // namespace detail
}  // namespace operators
}  // namespace paddle
