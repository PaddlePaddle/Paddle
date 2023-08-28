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

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_api.h"
#include "paddle/fluid/primitive/backend/manual/manual_backend.h"
#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"

namespace paddle {
namespace primitive {
namespace backend {

using LazyTensor = paddle::primitive::LazyTensor;

template <>
std::vector<Tensor> concat_grad<LazyTensor>(const std::vector<Tensor>& x,
                                            const Tensor& out_grad,
                                            const Tensor& axis) {
  std::vector<ir::OpResult> x_res;
  for (uint64_t idx = 0; idx < x.size(); idx++) {
    x_res.emplace_back(std::static_pointer_cast<LazyTensor>(x[idx].impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>());
  }

  ir::OpResult out_grad_res =
      std::static_pointer_cast<LazyTensor>(out_grad.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>();

  ir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis.impl())
                              ->getValue()
                              .dyn_cast<ir::OpResult>();

  std::vector<ir::OpResult> op_res =
      paddle::dialect::concat_grad(x_res, out_grad_res, axis_res);

  std::vector<Tensor> op_result;
  for (uint64_t idx = 0; idx < op_res.size(); idx++) {
    op_result.emplace_back(
        std::make_shared<primitive::LazyTensor>(op_res[idx]));
  }
  return op_result;
}

template <>
Tensor split_grad<LazyTensor>(const std::vector<Tensor>& out_grads,
                              const Tensor& axis) {
  std::vector<ir::OpResult> out_grads_res;
  for (uint64_t idx = 0; idx < out_grads.size(); idx++) {
    out_grads_res.emplace_back(
        std::static_pointer_cast<LazyTensor>(out_grads[idx].impl())
            ->getValue()
            .dyn_cast<ir::OpResult>());
  }
  ir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis.impl())
                              ->getValue()
                              .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::split_grad(out_grads_res, axis_res);
  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

}  // namespace backend
}  // namespace primitive
}  // namespace paddle
