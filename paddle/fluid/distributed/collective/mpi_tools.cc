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

#include "paddle/fluid/distributed/collective/mpi_tools.h"
#include "paddle/fluid/distributed/collective/Types.h"
#include "paddle/fluid/distributed/collective/common.h"

namespace paddle {
namespace distributed {
namespace mpi {

MPI_Op ToMPIType(ReduceOp reduction) {
  static const std::map<ReduceOp, MPI_Op> red_type = {
      {ReduceOp::MIN, MPI_MIN},
      {ReduceOp::MAX, MPI_MAX},
      {ReduceOp::SUM, MPI_SUM},
      {ReduceOp::PRODUCT, MPI_PROD},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(it != red_type.end(),
                    true,
                    platform::errors::InvalidArgument(
                        "Invalid mpi reduction. Must be MPI_MIN | MPI_MAX | "
                        "MPI_PROD | MPI_SUM."));
  return it->second;
}

// NOTE: MPI dose not support CUDA aware now.
bool CheckMpiCudaAware() { return false; }

void CheckValidInputs(const std::vector<phi::DenseTensor>& tensors) {
  PADDLE_ENFORCE_EQ(
      tensors.size() == 1,
      true,
      platform::errors::InvalidArgument("the inputs size of MPI must be 1!"));

  PADDLE_ENFORCE_EQ(CheckTensorsInCudaPlace(tensors) && !CheckMpiCudaAware(),
                    false,
                    platform::errors::InvalidArgument(
                        "Found CUDA Tensor. But CUDA-aware MPI not support!"));
}

}  //  namespace mpi
}  //  namespace distributed
}  //  namespace paddle
