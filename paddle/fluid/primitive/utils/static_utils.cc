// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"
#include "paddle/pir/core/operation.h"

namespace paddle {
namespace primitive {
template <>
void set_output<LazyTensor>(const paddle::Tensor& x_tmp, paddle::Tensor* x) {
  x->set_impl(x_tmp.impl());
}

template <>
void by_pass<LazyTensor>(const paddle::Tensor& x, paddle::Tensor* real_out) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::assign(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  set_output<LazyTensor>(out, real_out);
}

template <>
phi::IntArray construct_int_array_form_tensor<LazyTensor>(const Tensor& x) {
  phi::IntArray res;
  pir::Value x_value = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  if (x_value.dyn_cast<pir::OpResult>() &&
      x_value.dyn_cast<pir::OpResult>()
          .owner()
          ->isa<paddle::dialect::FullIntArrayOp>()) {
    res = std::move(phi::IntArray(paddle::dialect::GetInt64Vector(
        x_value.dyn_cast<pir::OpResult>()
            .owner()
            ->dyn_cast<paddle::dialect::FullIntArrayOp>()
            .attribute("value"))));
  } else if (x_value.type().isa<pir::VectorType>()) {
    size_t x_size = x_value.type().dyn_cast<pir::VectorType>().size();
    res = std::move(phi::IntArray(std::vector<int64_t>(x_size, -1)));
    res.SetFromTensor(true);
  } else if (x_value.type().isa<paddle::dialect::DenseTensorType>()) {
    common::DDim x_dim =
        x_value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
    size_t x_size = common::product(x_dim);
    if (common::contain_unknown_dim(x_dim)) {
      x_size = 1;
    }
    res = std::move(phi::IntArray(std::vector<int64_t>(x_size, -1)));
    res.SetFromTensor(true);
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Only support VectorType or DenseTensorType "
                                   "or AllocatedDenseTensorType"));
  }
  VLOG(0) << "res.FromTensor() " << res.FromTensor();
  return res;
}

/**
 * @brief set output with empty grads in pir.
 *
 *  In pir, we use None type to express
 *  that value is not available.
 *  Some outputs in vjp are marked as unnecessary
 *  by stop_gradient with True. Therefore the
 *  type of those outputs that are unnecessary will
 *  be set with None.
 *
 */
void SetEmptyGrad(const std::vector<std::vector<Tensor>>& outputs,
                  const std::vector<std::vector<bool>>& stop_gradients) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (size_t j = 0; j < outputs[i].size(); ++j) {
      if (stop_gradients[i][j]) {
        std::static_pointer_cast<primitive::LazyTensor>(outputs[i][j].impl())
            ->set_empty();
      }
    }
  }
}

std::vector<std::vector<Tensor>> ConstructVjpResultByStopGradients(
    const std::vector<std::vector<Tensor>>& outputs,
    const std::vector<std::vector<bool>>& stop_gradients) {
  SetEmptyGrad(outputs, stop_gradients);
  std::vector<std::vector<Tensor>> vjp_results(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    vjp_results[i].reserve(outputs[i].size());
    for (size_t j = 0; j < outputs[i].size(); ++j) {
      if (stop_gradients[i][j]) {
        // Use Tensor's impl is nullptr to indicate it has no gradient
        vjp_results[i].emplace_back(Tensor());
      } else {
        vjp_results[i].emplace_back(outputs[i][j]);
      }
    }
  }
  return vjp_results;
}

}  // namespace primitive
}  // namespace paddle
