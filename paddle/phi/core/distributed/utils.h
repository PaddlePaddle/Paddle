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
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace distributed {

inline phi::DenseTensor GetPartialTensor(const phi::DenseTensor& tensor,
                                         int64_t offset,
                                         int64_t numel) {
  phi::DenseTensor tensor_flattened;
  tensor_flattened.ShareDataWith(tensor);
  tensor_flattened.Resize({tensor.numel()});
  return tensor_flattened.Slice(offset, offset + numel);
}

inline void* GetPointerByOffset(void* raw_pointer,
                                size_t offset,
                                phi::DataType type) {
  if (type == phi::DataType::FLOAT32) {
    return reinterpret_cast<void*>(reinterpret_cast<float*>(raw_pointer) +
                                   offset);
  } else if (type == phi::DataType::FLOAT64) {
    return reinterpret_cast<void*>(reinterpret_cast<double*>(raw_pointer) +
                                   offset);
  } else if (type == phi::DataType::FLOAT16) {
    return reinterpret_cast<void*>(reinterpret_cast<int16_t*>(raw_pointer) +
                                   offset);
  } else if (type == phi::DataType::INT32) {
    return reinterpret_cast<void*>(reinterpret_cast<int32_t*>(raw_pointer) +
                                   offset);
  } else if (type == phi::DataType::INT64) {
    return reinterpret_cast<void*>(reinterpret_cast<int64_t*>(raw_pointer) +
                                   offset);
  } else if (type == phi::DataType::INT8) {
    return reinterpret_cast<void*>(reinterpret_cast<int8_t*>(raw_pointer) +
                                   offset);
  } else if (type == phi::DataType::UINT8) {
    return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(raw_pointer) +
                                   offset);
  } else if (type == phi::DataType::BOOL) {
    return reinterpret_cast<void*>(reinterpret_cast<bool*>(raw_pointer) +
                                   offset);
  } else if (type == phi::DataType::BFLOAT16) {
    return reinterpret_cast<void*>(reinterpret_cast<uint16_t*>(raw_pointer) +
                                   offset);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Datatype %s in NCCL is not supported.", type));
  }
  return nullptr;
}

inline void CheckSizeOnEachRank(const phi::DDim& tensor_dim,
                                const std::vector<int64_t>& size_on_each_rank,
                                int world_size) {
  int length_size_on_each_rank = size_on_each_rank.size();
  PADDLE_ENFORCE_EQ(
      length_size_on_each_rank,
      world_size,
      phi::errors::InvalidArgument(
          "The length of size_on_each_rank must be equal to world_size."));

  int64_t sum_size_on_each_rank =
      std::accumulate(size_on_each_rank.begin(), size_on_each_rank.end(), 0);
  PADDLE_ENFORCE_EQ(
      sum_size_on_each_rank,
      tensor_dim[0],
      phi::errors::InvalidArgument(
          "The sum of size_on_each_rank must be equal to tensor's dim[0]."));
}

enum class CommType : std::uint8_t {
  BROADCAST = 0,
  ALLREDUCE = 1,
  ALLREDUCE_SPARSE = 2,  // TODO(shenliang03): to support sparse in allreduce
  REDUCE = 3,
  ALLGATHER = 4,
  GATHER = 5,
  SCATTER = 6,
  REDUCE_SCATTER = 7,
  ALLTOALL = 8,
  SEND = 9,
  RECV = 10,
  BARRIER = 11,
  UNKNOWN = 100,
};

inline bool IsP2POP(CommType comm_type, bool is_batch_p2p = false) {
  if (is_batch_p2p) {
    return false;
  } else {
    return comm_type == CommType::SEND || comm_type == CommType::RECV;
  }
}

inline std::string CommTypeToString(CommType CommType) {
  switch (CommType) {
    case CommType::BROADCAST:
      return "BROADCAST";
    case CommType::ALLREDUCE:
      return "ALLREDUCE";
    case CommType::ALLREDUCE_SPARSE:
      return "ALLREDUCE_SPARSE";
    case CommType::REDUCE:
      return "REDUCE";
    case CommType::ALLGATHER:
      return "ALLGATHER";
    case CommType::GATHER:
      return "GATHER";
    case CommType::SCATTER:
      return "SCATTER";
    case CommType::REDUCE_SCATTER:
      return "REDUCE_SCATTER";
    case CommType::ALLTOALL:
      return "ALLTOALL";
    case CommType::SEND:
      return "SEND";
    case CommType::RECV:
      return "RECV";
    case CommType::BARRIER:
      return "BARRIER";
    case CommType::UNKNOWN:
      return "UNKNOWN";
    default:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

}  //  namespace distributed
}  // namespace phi
