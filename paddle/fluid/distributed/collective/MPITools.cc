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

#include "paddle/fluid/distributed/collective/MPITools.h"
#include "paddle/fluid/distributed/collective/Types.h"

namespace paddle {
namespace distributed {

MPI_Op ToMPIRedType(ReduceOp reduction) {
  static const std::map<ReduceOp, ncclRedOp_t> red_type = {
      {ReduceOp::MIN, MPI_MIN},
      {ReduceOp::MAX, MPI_MAX},
      {ReduceOp::SUM, MPI_SUM},
      {ReduceOp::PRODUCT, MPI_PROD},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(it != red_type.end(), true,
                    platform::errors::InvalidArgument(
                        "Invalid mpi reduction. Must be MPI_MIN | MPI_MAX | "
                        "MPI_PROD | MPI_SUM."));
  return it->second;
}

bool CheckMpiCudaAware() {
// Run time check
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (MPIX_Query_cuda_support() == 1) {
    return true;
  } else {
    return false;
  }
#else
  return false;
#endif
}

void CheckValidInputs(const std::vector<phi::DenseTensor>& tensors) {
  PADDLE_ENFORCE_EQ(
      tensors.size() == 1, true,
      platform::errors::InvalidArgument("the inputs size of MPI must be 1!"));

  PADDLE_ENFORCE_EQ(tensors[0].is_cuda() && !CheckMpiCudaAware(), false,
                    platform::errors::InvalidArgument(
                        "Found CUDA Tensor. But CUDA-aware MPI not support!"));
}

void CheckValidSizeAndType(const phi::DenseTensor& t_in,
                           const std::vector<phi::DenseTensor>& inputs) {
  CheckValidInputs(tensors);
  for (const auto& tensor : inputs) {
    PADDLE_ENFORCE_EQ(
        (tensor.numel() != t_in.numel()) || (tensor.dtype() != t_in.dtype()),
        true, platform::errors::InvalidArgument(
                  "Tensors are not same in data type or size!"));
  }
}

}  //  namespace distributed
}  //  namespace paddle
