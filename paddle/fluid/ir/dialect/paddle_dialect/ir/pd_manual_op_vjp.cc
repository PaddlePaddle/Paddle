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

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_attribute.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/fluid/primitive/rule/vjp/vjp.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/op_base.h"
#include "paddle/phi/common/int_array.h"

// TODO(wanghao107)
// this file will be generated in pd_op.cc

namespace paddle {
namespace dialect {
using IntArray = paddle::experimental::IntArray;

std::vector<std::vector<ir::OpResult>> SumOp::Vjp(
    ir::Operation* op,
    const std::vector<std::vector<ir::OpResult>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  SumOp op_obj = op->dyn_cast<SumOp>();
  Tensor x(std::make_shared<primitive::LazyTensor>(op_obj.x()));
  Tensor out_grad(std::make_shared<primitive::LazyTensor>(out_grads[0][0]));

  IntArray axis = op_obj.axis()
                      .GetDefiningOp()
                      ->attribute("value")
                      .dyn_cast<paddle::dialect::IntArrayAttribute>()
                      .data();
  bool keepdim = op->attribute("keepdim").dyn_cast<ir::BoolAttribute>().data();
  bool reduce_all = false;
  std::vector<std::vector<Tensor>> tensor_res = primitive::sum_vjp(
      x, out_grad, axis, keepdim, reduce_all, stop_gradients);
  std::vector<std::vector<ir::OpResult>> res(2, std::vector<ir::OpResult>(1));
  if (tensor_res[0][0].defined()) {
    res[0][0] =
        std::static_pointer_cast<primitive::LazyTensor>(tensor_res[0][0].impl())
            ->getValue()
            .dyn_cast<ir::OpResult>();
  }
  return res;
}

}  // namespace dialect
}  // namespace paddle
