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

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/primitive/rule/vjp/vjp.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/op_base.h"

// TODO(wanghao107)
// this file will be generated in pd_op.cc

namespace paddle {
namespace dialect {
using IntArray = paddle::experimental::IntArray;

std::vector<std::vector<pir::OpResult>> SumOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::OpResult>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  SumOp op_obj = op->dyn_cast<SumOp>();
  Tensor x(std::make_shared<primitive::LazyTensor>(op_obj.x()));
  Tensor out_grad(std::make_shared<primitive::LazyTensor>(out_grads[0][0]));

  Tensor axis = Tensor(std::make_shared<primitive::LazyTensor>(op_obj.axis()));

  bool keepdim = op->attribute("keepdim").dyn_cast<pir::BoolAttribute>().data();
  bool reduce_all = false;
  std::vector<std::vector<Tensor>> tensor_res = primitive::sum_vjp(
      x, out_grad, axis, keepdim, reduce_all, stop_gradients);
  std::vector<std::vector<pir::OpResult>> res(2, std::vector<pir::OpResult>(1));
  if (tensor_res[0][0].defined()) {
    res[0][0] =
        std::static_pointer_cast<primitive::LazyTensor>(tensor_res[0][0].impl())
            ->value()
            .dyn_cast<pir::OpResult>();
  }
  return res;
}

}  // namespace dialect
}  // namespace paddle
