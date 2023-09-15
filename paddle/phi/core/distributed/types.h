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

#include <chrono>
#include <cstdint>
#include <vector>

#include "paddle/phi/common/place.h"

namespace phi {
namespace distributed {

// TODO(shenliang03): To support AVG for reduce
// TODO(liyurui): remove this reduce op, use phi reduce op instead.
enum class ReduceOp : std::uint8_t { SUM = 0, MAX, MIN, PRODUCT, AVG };

struct AllreduceOptions {
  ReduceOp reduce_op = ReduceOp::SUM;
};

struct BroadcastOptions {
  int source_rank = 0;
  int source_root = 0;
};

struct BarrierOptions {
  int8_t device_id;
};

struct ReduceOptions {
  ReduceOp reduce_op = ReduceOp::SUM;
  int root_rank = 0;
};

struct ScatterOptions {
  int root_rank = 0;
};

struct GatherOptions {
  int root_rank = 0;
};

struct ReduceScatterOptions {
  ReduceOp reduce_op = ReduceOp::SUM;
};

}  //  namespace distributed
}  // namespace phi
