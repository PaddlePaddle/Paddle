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

#include "paddle/fluid/primitive/rule/vjp/vjp.h"
#include "paddle/fluid/ir/dialect/pd_api.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"
#include "paddle/fluid/primitive/backend/static_backend.h"
#include "paddle/fluid/primitive/rule/vjp/details.h"
#include "paddle/fluid/primitive/rule/vjp/utils.h"
#include "paddle/fluid/primitive/type/desc_tensor.h"
#include "paddle/ir/core/operation.h"
// TODO(wanghao107):
//  op's vjp will be auto generated.

namespace paddle {
namespace primitive {

std::vector<std::vector<paddle::Tensor>> tanh_vjp(
    const Tensor& out,
    const Tensor& grad_out,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res(
      1, std::vector<paddle::Tensor>(1));
  // get tanh_grad res.
  Tensor op_res = backend::tanh_grad<primitive::DescTensor>(out, grad_out);

  // set op stop_gradient info
  // TODO(wanghao107): Replace with more generic code.
  // Support set stop_gradients for all ops.
  ir::Operation* grad_op =
      std::static_pointer_cast<primitive::DescTensor>(op_res.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>()
          .owner();
  uint32_t num_res = grad_op->num_results();
  std::vector<ir::Attribute> ir_stop_gradients(num_res);
  for (size_t i = 0; i < num_res; i++) {
    if (stop_gradients[0][i]) {
      ir_stop_gradients[i] =
          ir::BoolAttribute::get(ir::IrContext::Instance(), true);
    } else {
      ir_stop_gradients[i] =
          ir::BoolAttribute::get(ir::IrContext::Instance(), false);
    }
  }
  grad_op->set_attribute(
      "stop_gradient",
      ir::ArrayAttribute::get(ir::IrContext::Instance(), ir_stop_gradients));

  // construct vjp result by op result and stop_gradients info
  if (!stop_gradients[0][0]) {
    vjp_res[0][0] = op_res;
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> mean_vjp(
    const Tensor& x,
    const Tensor& out_grad,
    const IntArray& axis,
    bool keepdim,
    bool reduce_all,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res(
      1, std::vector<paddle::Tensor>(1));
  // get mean_grad res.
  Tensor op_res = backend::mean_grad<primitive::DescTensor>(
      x, out_grad, axis, keepdim, reduce_all);

  // set op stop_gradient info
  // TODO(wanghao107): Replace with more generic code.
  // Support set stop_gradients for all ops.
  ir::Operation* grad_op =
      std::static_pointer_cast<primitive::DescTensor>(op_res.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>()
          .owner();
  uint32_t num_res = grad_op->num_results();
  std::vector<ir::Attribute> ir_stop_gradients(num_res);
  for (size_t i = 0; i < num_res; i++) {
    if (stop_gradients[0][i]) {
      ir_stop_gradients[i] =
          ir::BoolAttribute::get(ir::IrContext::Instance(), true);
    } else {
      ir_stop_gradients[i] =
          ir::BoolAttribute::get(ir::IrContext::Instance(), false);
    }
  }
  grad_op->set_attribute(
      "stop_gradient",
      ir::ArrayAttribute::get(ir::IrContext::Instance(), ir_stop_gradients));

  // construct vjp result by op result and stop_gradients info
  if (!stop_gradients[0][0]) {
    vjp_res[0][0] = op_res;
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> add_vjp(
    const Tensor& x,
    const Tensor& y,
    const Tensor& out_grad,
    int axis,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res(
      2, std::vector<paddle::Tensor>(1));
  // get add_grad res.
  std::tuple<Tensor, Tensor> op_res =
      backend::add_grad<primitive::DescTensor>(x, y, out_grad, axis);

  // set op stop_gradient info
  // TODO(wanghao107): Replace with more generic code.
  // Support set stop_gradients for all ops.
  ir::Operation* grad_op = std::static_pointer_cast<primitive::DescTensor>(
                               std::get<0>(op_res).impl())
                               ->getValue()
                               .dyn_cast<ir::OpResult>()
                               .owner();
  std::vector<ir::Attribute> ir_stop_gradients(2);
  for (size_t i = 0; i < 2; i++) {
    if (stop_gradients[i][0]) {
      ir_stop_gradients[i] =
          ir::BoolAttribute::get(ir::IrContext::Instance(), true);
    } else {
      ir_stop_gradients[i] =
          ir::BoolAttribute::get(ir::IrContext::Instance(), false);
    }
  }
  grad_op->set_attribute(
      "stop_gradient",
      ir::ArrayAttribute::get(ir::IrContext::Instance(), ir_stop_gradients));

  // construct vjp result by op result and stop_gradients info
  vjp_res[0][0] = !stop_gradients[0][0] ? std::get<0>(op_res) : vjp_res[0][0];
  vjp_res[1][0] = !stop_gradients[1][0] ? std::get<1>(op_res) : vjp_res[1][0];
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> divide_vjp(
    const Tensor& x,
    const Tensor& y,
    const Tensor& out,
    const Tensor& out_grad,
    int axis,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res(
      2, std::vector<paddle::Tensor>(1));
  if (!paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    // get divide_grad res.
    std::tuple<Tensor, Tensor> op_res =
        backend::divide_grad<primitive::DescTensor>(x, y, out, out_grad, axis);
    // construct vjp result by op result and stop_gradients info
    vjp_res[0][0] = !stop_gradients[0][0] ? std::get<0>(op_res) : vjp_res[0][0];
    vjp_res[1][0] = !stop_gradients[1][0] ? std::get<1>(op_res) : vjp_res[1][0];
  } else {
    // get divide_grad  prim mode res.
    Tensor* dx = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr;
    Tensor* dy = !stop_gradients[1][0] ? &vjp_res[1][0] : nullptr;
    details::divide_grad<DescTensor>(x, y, out, out_grad, axis, dx, dy);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sum_vjp(
    const Tensor& x,
    const Tensor& out_grad,
    const IntArray& axis,
    bool keepdim,
    bool reduce_all,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res(
      1, std::vector<paddle::Tensor>(1));
  if (!paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    // get sum_grad res.
    Tensor op_res = backend::sum_grad<primitive::DescTensor>(
        x, out_grad, axis, keepdim, reduce_all);
    // construct vjp result by op result and stop_gradients info
    if (!stop_gradients[0][0]) {
      vjp_res[0][0] = op_res;
    }
  } else {
    // get divide_grad  prim mode res.
    Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr;
    details::sum_grad<DescTensor>(
        x, out_grad, axis, keepdim, reduce_all, x_grad);
  }
  return vjp_res;
}

}  // namespace primitive
}  // namespace paddle
