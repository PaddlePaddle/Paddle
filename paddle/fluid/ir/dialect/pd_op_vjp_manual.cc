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

#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/fluid/primitive/rule/vjp/vjp.h"
#include "paddle/fluid/primitive/type/desc_tensor.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/op_base.h"
#include "paddle/phi/common/int_array.h"

// TODO(wanghao107)
// this file will be generated in pd_op.cc

namespace paddle {
namespace dialect {
using IntArray = paddle::experimental::IntArray;

std::vector<std::vector<ir::OpResult>> TanhOp::Vjp(
    ir::Operation* op,
    const std::vector<std::vector<ir::OpResult>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  TanhOp op_obj = op->dyn_cast<TanhOp>();
  Tensor out(
      std::make_shared<primitive::experimental::DescTensor>(op_obj.out()));
  Tensor grad_out(
      std::make_shared<primitive::experimental::DescTensor>(out_grads[0][0]));
  std::vector<std::vector<Tensor>> tensor_res =
      primitive::experimental::tanh_vjp(out, grad_out, stop_gradients);
  std::vector<std::vector<ir::OpResult>> res(1, std::vector<ir::OpResult>(1));
  if (tensor_res[0][0].defined()) {
    res[0][0] = std::static_pointer_cast<primitive::experimental::DescTensor>(
                    tensor_res[0][0].impl())
                    ->getValue()
                    .dyn_cast<ir::OpResult>();
  }
  return res;
}

std::vector<std::vector<ir::OpResult>> Tanh_Op::Vjp(
    ir::Operation* op,
    const std::vector<std::vector<ir::OpResult>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  // TODO(wanghao107)
  // we don't support inplace now,
  // so use the non-inplace version instead currently.
  // Support inplace in the future.
  Tanh_Op op_obj = op->dyn_cast<Tanh_Op>();
  Tensor out(
      std::make_shared<primitive::experimental::DescTensor>(op_obj.out()));
  Tensor grad_out(
      std::make_shared<primitive::experimental::DescTensor>(out_grads[0][0]));
  std::vector<std::vector<Tensor>> tensor_res =
      primitive::experimental::tanh_vjp(out, grad_out, stop_gradients);
  std::vector<std::vector<ir::OpResult>> res(1, std::vector<ir::OpResult>(1));
  if (tensor_res[0][0].defined()) {
    res[0][0] = std::static_pointer_cast<primitive::experimental::DescTensor>(
                    tensor_res[0][0].impl())
                    ->getValue()
                    .dyn_cast<ir::OpResult>();
  }
  return res;
}

std::vector<std::vector<ir::OpResult>> MeanOp::Vjp(
    ir::Operation* op,
    const std::vector<std::vector<ir::OpResult>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  MeanOp op_obj = op->dyn_cast<MeanOp>();
  Tensor x(std::make_shared<primitive::experimental::DescTensor>(op_obj.x()));
  Tensor out_grad(
      std::make_shared<primitive::experimental::DescTensor>(out_grads[0][0]));

  IntArray axis = op->attribute("axis")
                      .dyn_cast<paddle::dialect::IntArrayAttribute>()
                      .data();
  bool keepdim = op->attribute("keepdim").dyn_cast<ir::BoolAttribute>().data();
  bool reduce_all = false;
  std::vector<std::vector<Tensor>> tensor_res =
      primitive::experimental::mean_vjp(
          x, out_grad, axis, keepdim, reduce_all, stop_gradients);
  std::vector<std::vector<ir::OpResult>> res(1, std::vector<ir::OpResult>(1));
  if (tensor_res[0][0].defined()) {
    res[0][0] = std::static_pointer_cast<primitive::experimental::DescTensor>(
                    tensor_res[0][0].impl())
                    ->getValue()
                    .dyn_cast<ir::OpResult>();
  }
  return res;
}

std::vector<std::vector<ir::OpResult>> ConcatOp::Vjp(
    ir::Operation* op,
    const std::vector<std::vector<ir::OpResult>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  ConcatOp op_obj = op->dyn_cast<ConcatOp>();
  ir::CombineOp combine_op_obj =
      op_obj.x().GetDefiningOp()->dyn_cast<ir::CombineOp>();
  std::vector<Tensor> x;
  for (size_t idx = 0; idx < combine_op_obj.inputs().size(); idx++) {
    x.emplace_back(std::make_shared<primitive::experimental::DescTensor>(
        combine_op_obj.inputs()[idx]));
  }

  Tensor out_grad(
      std::make_shared<primitive::experimental::DescTensor>(out_grads[0][0]));

  Tensor axis(
      std::make_shared<primitive::experimental::DescTensor>(op_obj.axis()));

  std::vector<std::vector<Tensor>> tensor_res =
      primitive::experimental::concat_vjp(x, out_grad, axis, stop_gradients);
  std::vector<std::vector<ir::OpResult>> res(1, std::vector<ir::OpResult>(1));
  if (tensor_res[0][0].defined()) {
    res[0][0] = std::static_pointer_cast<primitive::experimental::DescTensor>(
                    tensor_res[0][0].impl())
                    ->getValue()
                    .dyn_cast<ir::OpResult>();
  }
  return res;
}

}  // namespace dialect
}  // namespace paddle
