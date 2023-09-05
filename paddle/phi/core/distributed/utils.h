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
