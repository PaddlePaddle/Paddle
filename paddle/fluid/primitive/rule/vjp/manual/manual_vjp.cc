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

// Auto Generated, DO NOT EDIT!

#include "paddle/fluid/primitive/rule/vjp/manual/manual_vjp.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"
#include "paddle/fluid/primitive/backend/backend.h"
#include "paddle/fluid/primitive/rule/vjp/details.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"
#include "paddle/pir/core/operation.h"

namespace paddle {
namespace primitive {

std::vector<std::vector<paddle::Tensor>> add_n_vjp(
    const std::vector<paddle::Tensor>& x,
    const Tensor& out_grad,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg : stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::add_n_grad<LazyTensor>(x, out_grad);
  vjp_res[0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> reshape_vjp(
    const Tensor& xshape,
    const Tensor& out_grad,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg : stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "reshape_grad";
  auto need_skip =
      paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(
          op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() &&
      !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call Pir Decomposed backward op reshape_grad";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr;

    details::reshape_grad<LazyTensor>(xshape, out_grad, x_grad);
  } else {
    auto op_res = backend::reshape_grad<LazyTensor>(xshape, out_grad);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

}  // namespace primitive
}  // namespace paddle
