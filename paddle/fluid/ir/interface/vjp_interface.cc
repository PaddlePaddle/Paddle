// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/primitive/rule/vjp/vjp_dispatch.h"

namespace paddle {
namespace dialect {
std::vector<std::vector<ir::OpResult>> TanhOp::Vjp(
    std::vector<std::vector<ir::OpResult>> out_grads,
    const std::vector<std::vector<int>>& stop_gradients) {
  return {{}};
}

std::vector<std::vector<ir::OpResult>> Tanh_Op::Vjp(
    std::vector<std::vector<ir::OpResult>> out_grads,
    const std::vector<std::vector<int>>& stop_gradients) {
  return {{}};
}
}  // namespace dialect
}  // namespace paddle
