/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <mpi.h>

#include <mutex>  // NOLINT
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag mpi_dso_flag;
extern void* mpi_dso_handle;

#define DECLARE_DYNAMIC_LOAD_MPI_WRAP(__name)                    \
  struct DynLoad__##__name {                                     \
    template <typename... Args>                                  \
    auto operator()(Args... args) -> decltype(__name(args...)) { \
      using mpi_func = decltype(&::__name);                      \
      std::call_once(mpi_dso_flag, []() {                        \
        mpi_dso_handle = phi::dynload::GetMPIDsoHandle();        \
      });                                                        \
      static void* p_##__name = dlsym(mpi_dso_handle, #__name);  \
      return reinterpret_cast<mpi_func>(p_##__name)(args...);    \
    }                                                            \
  };                                                             \
  extern DynLoad__##__name __name

#define MPI_RAND_ROUTINE_EACH(__macro) \
  __macro(MPI_Initialized);            \
  __macro(MPI_Comm_size);              \
  __macro(MPI_Comm_rank);              \
  __macro(MPI_Intercomm_merge);        \
  __macro(MPI_Intercomm_create);       \
  __macro(MPI_Comm_free);              \
  __macro(MPI_Init_thread);            \
  __macro(MPI_Comm_connect);           \
  __macro(MPI_Comm_split);             \
  __macro(MPI_Allreduce);              \
  __macro(MPI_Allgather);              \
  __macro(MPI_Reduce);                 \
  __macro(MPI_Bcast);                  \
  __macro(MPI_Barrier);                \
  __macro(MPI_Alltoall);               \
  __macro(MPI_Scatter);                \
  __macro(MPI_Isend);                  \
  __macro(MPI_Irecv);                  \
  __macro(MPI_Send);                   \
  __macro(MPI_Recv);                   \
  __macro(MPI_Finalized);

MPI_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MPI_WRAP)

}  // namespace dynload
}  // namespace phi
