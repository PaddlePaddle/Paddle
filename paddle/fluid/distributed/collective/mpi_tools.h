// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <error.h>
#include <iostream>
#include <string>
#include "paddle/phi/core/enforce.h"

#include "paddle/fluid/distributed/collective/types.h"

#ifdef HOST
#undef HOST
#endif

#include <mpi.h>

namespace paddle {
namespace distributed {
namespace mpi {

#define MPI_CHECK(cmd)                                             \
  do {                                                             \
    int r = cmd;                                                   \
    if (r != MPI_SUCCESS) {                                        \
      std::stringstream ss;                                        \
      ss << "Failed, MPI error in" << __FILE__ << ":" << __LINE__  \
         << "with error code: " << std::to_string(r) << std::endl; \
      PADDLE_THROW(common::errors::Fatal(ss.str()));               \
      exit(EXIT_FAILURE);                                          \
    }                                                              \
  } while (0)

MPI_Op ToMPIType(ReduceOp reduction);

bool CheckMpiCudaAware();

void CheckValidInputs(const std::vector<phi::DenseTensor>& tensors);

}  // namespace mpi
}  // namespace distributed
}  // namespace paddle
