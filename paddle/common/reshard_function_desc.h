// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace distributed {

struct BaseOpDesc {
  BaseOpDesc(const std::string& n, DataType d) : name(n), dtype(d) {}
  BaseOpDesc(const std::string& n, DataType d, const std::vector<int64_t>& pids)
      : name(n), dtype(d), process_ids(pids) {}
  virtual ~BaseOpDesc() {}

  std::string name;
  DataType dtype;
  std::vector<int64_t> process_ids;
};

struct AllReduceOpDesc : public BaseOpDesc {
  AllReduceOpDesc(DataType dt, const std::vector<int64_t>& pids, int red_type)
      : BaseOpDesc("AllReduce", dt, pids), reduce_type(red_type) {}
  int reduce_type;
};

struct SendOpDesc : public BaseOpDesc {
  SendOpDesc(DataType dt,
             const std::vector<int64_t>& pids,
             int peer_rank,
             bool dyn_shape)
      : BaseOpDesc("Send", dt, pids),
        peer(peer_rank),
        dynamic_shape(dyn_shape) {}

  int peer;
  bool dynamic_shape;
};

struct RecvOpDesc : public BaseOpDesc {
  RecvOpDesc(DataType dt,
             const std::vector<int64_t>& pids,
             int peer_rank,
             bool dyn_shape)
      : BaseOpDesc("Recv", dt, pids),
        peer(peer_rank),
        dynamic_shape(dyn_shape) {}

  int peer;
  bool dynamic_shape;
};

}  // namespace distributed
}  // namespace phi
