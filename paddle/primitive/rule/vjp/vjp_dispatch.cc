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

#include <math.h>
#include <vector>

#include "paddle/ir/core/value.h"
#include "paddle/primitive/ir_api/ir_api.h"
#include "paddle/primitive/rule/vjp/vjp_dispatch.h"
#include "paddle/primitive/type/desc_tensor.h"

namespace paddle {
namespace primitive {
namespace experimental {
std::vector<std::vector<Tensor>> tanh_vjp(
    const Tensor& out,
    const Tensor& grad_out,
    const std::vector<std::vector<int>>& stop_gradients) {
  // 1.constuct out and grad_out OpResult
  std::vector<std::vector<Tensor>> res;
  ir::OpResult out_opres = std::static_pointer_cast<DescTensor>(out.impl())
                               ->getValue()
                               .dyn_cast<ir::OpResult>();
  ir::OpResult grad_out_opres =
      std::static_pointer_cast<DescTensor>(grad_out.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>();

  // 2.call tanh_grad api
  ir::api::tanh_grad(out_opres, grad_out_opres);

  // 3.set stop_gradient info

  // 4.construct result by stop_gradients

  return res;
}
}  // namespace experimental
}  // namespace primitive
}  // namespace paddle
