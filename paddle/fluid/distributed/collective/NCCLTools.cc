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

#include "paddle/fluid/distributed/collective/NCCLTools.h"

#include "paddle/fluid/distributed/collective/Types.h"

namespace paddle {
namespace distributed {

ncclRedOp_t ToNCCLRedType(ReduceOp reduction) {
  static const std::map<ReduceOp, ncclRedOp_t> red_type = {
      {ReduceOp::MIN, ncclMin},
      {ReduceOp::MAX, ncclMax},
      {ReduceOp::SUM, ncclSum},
      {ReduceOp::PRODUCT, ncclProd},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(it != red_type.end(),
                    true,
                    platform::errors::InvalidArgument(
                        "Invalid nccl reduction. Must be ncclMin | ncclMax | "
                        "ncclProd | ncclSum"));
  return it->second;
}

std::string SerializeNCCLUniqueId(const ncclUniqueId& ncclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
  std::ostringstream oss;
  for (auto i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

void StaticCheckTensor(const phi::DenseTensor& tensor,
                       int rank,
                       int world_size) {
  // place check
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(tensor.place()),
      true,
      platform::errors::InvalidArgument("Tensor should be in GPU place."));
  // rank check
  PADDLE_ENFORCE_GE(rank,
                    0,
                    platform::errors::InvalidArgument(
                        "Rank should be greater than or equal to 0."));
  PADDLE_ENFORCE_LT(
      rank,
      world_size,
      platform::errors::InvalidArgument("Rank is out of the process group."));
}

// static check for collective
void StaticCheckTensors(const phi::DenseTensor& out_tensor,
                        const phi::DenseTensor& in_tensor,
                        int rank,
                        int world_size,
                        int out_size_factor,
                        int in_size_factor) {
  // place check
  PADDLE_ENFORCE_EQ(platform::is_gpu_place(out_tensor.place()),
                    true,
                    platform::errors::InvalidArgument(
                        "Output tensor should be in GPU place."));
  PADDLE_ENFORCE_EQ(platform::is_gpu_place(in_tensor.place()),
                    true,
                    platform::errors::InvalidArgument(
                        "Input tensor should be in GPU place."));
  // rank check
  PADDLE_ENFORCE_GE(rank,
                    0,
                    platform::errors::InvalidArgument(
                        "Rank should be greater than or equal to 0."));
  PADDLE_ENFORCE_LT(
      rank,
      world_size,
      platform::errors::InvalidArgument("Rank is out of the process group."));
  // shape check
  int64_t out_size = out_tensor.numel();
  PADDLE_ENFORCE_GT(out_size,
                    0,
                    platform::errors::InvalidArgument(
                        "Size of output tensor should be greater than 0."));
  int64_t in_size = in_tensor.numel();
  PADDLE_ENFORCE_GT(in_size,
                    0,
                    platform::errors::InvalidArgument(
                        "Size of input tensor should be greater than 0."));
  PADDLE_ENFORCE_EQ(
      out_size * out_size_factor,
      in_size * in_size_factor,
      platform::errors::InvalidArgument(
          "Input and output tensors should have matching sizes."));
  // dtype check
  PADDLE_ENFORCE_EQ(
      out_tensor.dtype(),
      in_tensor.dtype(),
      platform::errors::InvalidArgument(
          "Input and output tensors should have the same data type."));
}

void StaticCheckTensorsSameShape(const phi::DenseTensor& out_tensor,
                                 const phi::DenseTensor& in_tensor,
                                 int rank,
                                 int world_size) {
  StaticCheckTensors(out_tensor,
                     in_tensor,
                     rank,
                     world_size,
                     /*out_size_factor*/ 1,
                     /*in_size_factor*/ 1);
}

void StaticCheckTensorsScatterLikeShape(const phi::DenseTensor& out_tensor,
                                        const phi::DenseTensor& in_tensor,
                                        int rank,
                                        int world_size) {
  StaticCheckTensors(out_tensor,
                     in_tensor,
                     rank,
                     world_size,
                     /*out_size_factor*/ world_size,
                     /*in_size_factor*/ 1);
}

void StaticCheckTensorsGatherLikeShape(const phi::DenseTensor& out_tensor,
                                       const phi::DenseTensor& in_tensor,
                                       int rank,
                                       int world_size) {
  StaticCheckTensors(out_tensor,
                     in_tensor,
                     rank,
                     world_size,
                     /*out_size_factor*/ 1,
                     /*in_size_factor*/ world_size);
}

}  //  namespace distributed
}  //  namespace paddle
