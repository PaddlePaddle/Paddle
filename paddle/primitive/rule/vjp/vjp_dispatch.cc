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

#include "paddle/fluid/ir/dialect/ir_api.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value.h"
#include "paddle/primitive/rule/vjp/vjp_dispatch.h"
#include "paddle/primitive/type/desc_tensor.h"

namespace paddle {
namespace primitive {
namespace experimental {
std::vector<std::vector<Tensor>> tanh_vjp(
    const Tensor& out,
    const Tensor& grad_out,
    const std::vector<int>& stop_gradients) {
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
  std::vector<ir::OpResult> op_res =
      ir::api::tanh_grad(out_opres, grad_out_opres);

  // 3.set op stop_gradient info
  ir::Operation* grad_op_ptr = op_res[0].owner();
  uint32_t num_res = grad_op_ptr->num_results();
  std::vector<ir::Attribute> ir_stop_gradients(num_res);
  for (size_t i = 0; i < num_res; i++) {
    if (stop_gradients[i]) {
      ir_stop_gradients[i] =
          ir::BoolAttribute::get(ir::IrContext::Instance(), true);
    } else {
      ir_stop_gradients[i] =
          ir::BoolAttribute::get(ir::IrContext::Instance(), false);
    }
  }
  grad_op_ptr->set_attribute(
      "stop_gradient",
      ir::ArrayAttribute::get(ir::IrContext::Instance(), ir_stop_gradients));

  // 4.construct result by stop_gradients
  res.reserve(num_res);
  for (size_t i = 0; i < stop_gradients.size(); i++) {
    // TODO(wanghao107): maybe slice here
    res.emplace_back(std::vector<Tensor>{Tensor(
        std::make_shared<primitive::experimental::DescTensor>(op_res[i]))});
  }
  return res;
}
}  // namespace experimental
}  // namespace primitive
}  // namespace paddle
