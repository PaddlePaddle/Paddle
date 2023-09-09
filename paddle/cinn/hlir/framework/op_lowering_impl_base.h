// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/cinn/ir/lowered_func.h"

// Fusion Op lowering, there are four kinds of lowering function:
// Elementwise/Broadcast/Injective,Reduce,OutEWiseFusable,NonFusible.
// Elementwise/Broadcast/Injective Ops is with same shcedule.
// Reduce,OutEWiseFusable,NonFusible are using different schedule.

namespace cinn {
namespace hlir {
namespace framework {

template <typename T>
class OpLowererImplBase {
 public:
  OpLowererImplBase() = default;
  ~OpLowererImplBase() = default;

  virtual std::vector<ir::LoweredFunc> Lower(
      const T& group,
      bool apply_op_schedule = true,
      bool apply_group_schedule = true) = 0;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
