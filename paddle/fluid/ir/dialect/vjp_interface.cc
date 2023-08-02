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
#include "paddle/ir/core/op_base.h"
#include "paddle/primitive/rule/vjp/vjp_dispatch.h"
#include "paddle/primitive/type/desc_tensor.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace dialect {
std::vector<std::vector<ir::OpResult>> TanhOp::Vjp(
    ir::Operation* op,
    const std::vector<std::vector<ir::OpResult>>& out_grads,
    const std::vector<std::vector<int>>& stop_gradients) {
  TanhOp op_obj = op->dyn_cast<TanhOp>();
  Tensor out(
      std::make_shared<primitive::experimental::DescTensor>(op_obj.out()));
  Tensor grad_out(
      std::make_shared<primitive::experimental::DescTensor>(out_grads[0][0]));
  paddle::optional<paddle::Tensor> tensor_res =
      primitive::experimental::tanh_vjp(out, grad_out, stop_gradients);
  std::vector<std::vector<ir::OpResult>> res(1, std::vector<ir::OpResult>(1));
  if (!stop_gradients[0][0]) {
    res[0][0] = std::static_pointer_cast<primitive::experimental::DescTensor>(
                    tensor_res.get().impl())
                    ->getValue()
                    .dyn_cast<ir::OpResult>();
  }
  return res;
}

std::vector<std::vector<ir::OpResult>> Tanh_Op::Vjp(
    ir::Operation* op,
    const std::vector<std::vector<ir::OpResult>>& out_grads,
    const std::vector<std::vector<int>>& stop_gradients) {
  return {};
}
}  // namespace dialect
}  // namespace paddle
