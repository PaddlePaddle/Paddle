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
#include "paddle/fluid/primitive/base/lazy_tensor.h"
#include "paddle/fluid/primitive/primitive/primitive.h"

namespace paddle::primitive::backend {

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
  std::transform(
      op_res.begin(), op_res.end(), x_grad.begin(), [](const pir::Value& res) {
        return Tensor(std::make_shared<LazyTensor>(res));
      });
  return x_grad;
}

template <>
Tensor embedding_grad<LazyTensor>(const Tensor& x,
                                  const Tensor& weight,
                                  const Tensor& out_grad,
                                  int64_t padding_idx,
                                  bool sparse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value weight_res =
      std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  pir::Value out_grad_res =
      std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::embedding_grad(
      x_res, weight_res, out_grad_res, padding_idx, sparse);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out;
}

template <>
std::tuple<Tensor, Tensor, Tensor> fused_gemm_epilogue_grad<LazyTensor>(
    const Tensor& x,
    const Tensor& y,
    const paddle::optional<Tensor>& reserve_space,
    const Tensor& out_grad,
    bool trans_x,
    bool trans_y,
    const std::string& activation) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  paddle::optional<pir::Value> reserve_space_res;
  if (reserve_space) {
    pir::Value reserve_space_res_inner;
    reserve_space_res_inner =
        std::static_pointer_cast<LazyTensor>(reserve_space.get().impl())
            ->value();
    reserve_space_res =
        paddle::make_optional<pir::Value>(reserve_space_res_inner);
  }
  pir::Value out_grad_res =
      std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fused_gemm_epilogue_grad(x_res,
                                                          y_res,
                                                          reserve_space_res,
                                                          out_grad_res,
                                                          trans_x,
                                                          trans_y,
                                                          activation);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, y_grad, bias_grad);
}

}  // namespace paddle::primitive::backend
