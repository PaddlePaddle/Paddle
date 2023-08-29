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
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_api.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"
#include "paddle/fluid/primitive/backend/backend.h"
#include "paddle/fluid/primitive/rule/vjp/details.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"
#include "paddle/ir/core/operation.h"

namespace paddle {
namespace primitive {

std::vector<std::vector<paddle::Tensor>> concat_vjp(
    const std::vector<Tensor>& x,
    const Tensor& out_grad,
    const Tensor& axis,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res(2, std::vector<Tensor>());
  // get concat_grad res.
  std::vector<Tensor> op_res =
      backend::concat_grad<primitive::LazyTensor>(x, out_grad, axis);

  // construct vjp result by op result and stop_gradients info
  vjp_res[0].resize(op_res.size());
  for (uint64_t idx = 0; idx < op_res.size(); idx++) {
    if (!stop_gradients[0][idx]) {
      vjp_res[0][idx] = op_res[idx];
    }
  }
  // vjp_res[1] is axis's grad which is attribute (no grad).
  vjp_res[1].resize(1);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> split_vjp(
    const std::vector<Tensor>& out_grads,
    const Tensor& axis,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res(3, std::vector<Tensor>(1));
  // get concat_grad res.
  Tensor op_res = backend::split_grad<primitive::LazyTensor>(out_grads, axis);

  // construct vjp result by op result and stop_gradients info
  if (!stop_gradients[0][0]) {
    vjp_res[0][0] = op_res;
  }

  // vjp_res[1] is sections's grad which is attribute (no grad).
  // vjp_res[2] is axis's grad which is attribute (no grad).
  vjp_res[1].resize(stop_gradients[1].size());
  vjp_res[2].resize(stop_gradients[2].size());
  return vjp_res;
}

}  // namespace primitive
}  // namespace paddle
