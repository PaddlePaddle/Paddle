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
#include <sstream>
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
      return "Broadcast";
    case CommType::ALLREDUCE:
      return "AllReduce";
    case CommType::ALLREDUCE_SPARSE:
      return "AllReduce_Sparse";
    case CommType::REDUCE:
      return "Reduce";
    case CommType::ALLGATHER:
      return "AllGather";
    case CommType::GATHER:
      return "Gather";
    case CommType::SCATTER:
      return "Scatter";
    case CommType::REDUCE_SCATTER:
      return "ReduceScatter";
    case CommType::ALLTOALL:
      return "AllToAll";
    case CommType::SEND:
      return "Send";
    case CommType::RECV:
      return "Recv";
    case CommType::BARRIER:
      return "Barrier";
    case CommType::UNKNOWN:
      return "Unknown";
    default:
      return "Unknown";
  }
  return "Unknown";
}

// convert vector to string, concatenate continuous intervals with `~`,
// concatenate discontinuous intervals with `#` eg: [1,2,3,4,5,7,8,9] =>
// 1~3#4#5#7~9
inline std::string VectorToString(const std::vector<int>& vec) {
  if (vec.empty()) {
    return "";
  }
  if (vec.size() == 1) {
    return std::to_string(vec[0]);
  }

  std::stringstream ss;
  size_t i = 0;
  int start_rank = vec[i];
  for (; i < vec.size() - 1; ++i) {
    if (vec[i] + 1 == vec[i + 1]) {
      continue;
    }
    if (ss.rdbuf()->in_avail() != 0) {
      ss << "#";
    }
    ss << start_rank;
    if (start_rank != vec[i]) {
      ss << "~";
      ss << vec[i];
    }
    start_rank = vec[i + 1];
  }

  if (ss.rdbuf()->in_avail() != 0) {
    ss << "#";
  }
  ss << start_rank;
  if (start_rank != vec[i]) {
    ss << "~";
    ss << vec[i];
  }

  return ss.str();
}

inline void LogLongStr(const std::string prefix, const std::string& log) {
  size_t max_log_size = 20000;
  if (log.size() >= max_log_size) {
    int log_count = log.size() / max_log_size + 1;
    int index = 0;
    int part = 0;
    while (index + max_log_size < log.size()) {
      LOG(INFO) << prefix << "part:" << part << "/" << log_count << ","
                << log.substr(index, max_log_size) << std::endl;
      index += max_log_size;
      part++;
    }
    LOG(INFO) << prefix << "part:" << part << "/" << log_count << ","
              << log.substr(index) << std::endl;
  } else {
    LOG(INFO) << prefix << "part:0/1," << log << std::endl;
  }
}

}  //  namespace distributed
}  // namespace phi
