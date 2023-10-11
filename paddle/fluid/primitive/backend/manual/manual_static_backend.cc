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

#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/primitive/backend/generated/generated_backend.h"
#include "paddle/fluid/primitive/backend/manual/manual_backend.h"
#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"

namespace paddle {
namespace primitive {
namespace backend {

using LazyTensor = paddle::primitive::LazyTensor;
template <>
std::vector<Tensor> add_n_grad<LazyTensor>(const std::vector<Tensor>& x,
                                           const Tensor& out_grad) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value out_grad_res =
      std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::add_n_grad(x_res, out_grad_res);

  std::vector<Tensor> x_grad(op_res.size());
  std::transform(op_res.begin(),
                 op_res.end(),
                 x_grad.begin(),
                 [](const pir::OpResult& res) {
                   return Tensor(std::make_shared<LazyTensor>(res));
                 });
  return x_grad;
}

}  // namespace backend
}  // namespace primitive
}  // namespace paddle
