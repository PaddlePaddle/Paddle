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

#include "paddle/fluid/primitive/backend/static_backend.h"
#include "paddle/fluid/ir/dialect/pd_api.h"
#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"

namespace paddle {
namespace primitive {
namespace backend {

using LazyTensor = paddle::primitive::LazyTensor;

template <>
Tensor tanh_grad<LazyTensor>(const Tensor& out, const Tensor& grad_out) {
  ir::OpResult out_res = std::static_pointer_cast<LazyTensor>(out.impl())
                             ->getValue()
                             .dyn_cast<ir::OpResult>();
  ir::OpResult grad_out_res =
      std::static_pointer_cast<LazyTensor>(grad_out.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>();

  ir::OpResult op_res = paddle::dialect::tanh_grad(out_res, grad_out_res);

  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
Tensor mean_grad<LazyTensor>(const Tensor& x,
                             const Tensor& out_grad,
                             const IntArray& axis,
                             bool keepdim,
                             bool reduce_all) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult out_grad_res =
      std::static_pointer_cast<LazyTensor>(out_grad.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>();

  ir::OpResult op_res = paddle::dialect::mean_grad(
      x_res, out_grad_res, axis.GetData(), keepdim, reduce_all);

  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
Tensor divide<LazyTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::divide(x_res, y_res);
  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
Tensor add<LazyTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::add(x_res, y_res);
  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
Tensor multiply<LazyTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::multiply(x_res, y_res);
  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
Tensor elementwise_pow<LazyTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::elementwise_pow(x_res, y_res);
  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
Tensor scale<LazyTensor>(const Tensor& x,
                         const Scalar& scale,
                         float bias,
                         bool bias_after_scale) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res =
      paddle::dialect::scale(x_res, scale.to<float>(), bias, bias_after_scale);
  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
Tensor sum<LazyTensor>(const Tensor& x,
                       const IntArray& axis,
                       phi::DataType dtype,
                       bool keepdim) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res =
      paddle::dialect::sum(x_res, axis.GetData(), dtype, keepdim);
  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
Tensor full<LazyTensor>(const IntArray& shape,
                        const Scalar& value,
                        phi::DataType dtype,
                        phi::Place place) {
  ir::OpResult op_res =
      paddle::dialect::full(shape.GetData(), value.to<float>(), dtype, place);
  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
std::tuple<Tensor, Tensor> reshape<LazyTensor>(const Tensor& x,
                                               const IntArray& shape) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  std::tuple<ir::OpResult, ir::OpResult> op_res =
      paddle::dialect::reshape(x_res, shape.GetData());
  return std::make_tuple(
      Tensor(std::make_shared<primitive::LazyTensor>(std::get<0>(op_res))),
      Tensor(std::make_shared<primitive::LazyTensor>(std::get<1>(op_res))));
}

template <>
Tensor expand<LazyTensor>(const Tensor& x, const IntArray& shape) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::expand(x_res, shape.GetData());
  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
Tensor tile<LazyTensor>(const Tensor& x, const IntArray& repeat_times) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::tile(x_res, repeat_times.GetData());
  return Tensor(std::make_shared<primitive::LazyTensor>(op_res));
}

template <>
std::tuple<Tensor, Tensor> add_grad<LazyTensor>(const Tensor& x,
                                                const Tensor& y,
                                                const Tensor& out_grad,
                                                int axis) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult out_grad_res =
      std::static_pointer_cast<LazyTensor>(out_grad.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>();

  std::tuple<ir::OpResult, ir::OpResult> op_res =
      paddle::dialect::add_grad(x_res, y_res, out_grad_res, axis);

  return std::make_tuple(
      Tensor(std::make_shared<primitive::LazyTensor>(std::get<0>(op_res))),
      Tensor(std::make_shared<primitive::LazyTensor>(std::get<1>(op_res))));
}

template <>
std::tuple<Tensor, Tensor> divide_grad<LazyTensor>(const Tensor& x,
                                                   const Tensor& y,
                                                   const Tensor& out,
                                                   const Tensor& out_grad,
                                                   int axis) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult out_res = std::static_pointer_cast<LazyTensor>(out.impl())
                             ->getValue()
                             .dyn_cast<ir::OpResult>();
  ir::OpResult out_grad_res =
      std::static_pointer_cast<LazyTensor>(out_grad.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>();

  std::tuple<ir::OpResult, ir::OpResult> op_res =
      paddle::dialect::divide_grad(x_res, y_res, out_res, out_grad_res, axis);

  return std::make_tuple(
      Tensor(std::make_shared<LazyTensor>(std::get<0>(op_res))),
      Tensor(std::make_shared<LazyTensor>(std::get<1>(op_res))));
}

template <>
Tensor sum_grad<LazyTensor>(const Tensor& x,
                            const Tensor& out_grad,
                            const IntArray& axis,
                            bool keepdim,
                            bool reduce_all) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult out_grad_res =
      std::static_pointer_cast<LazyTensor>(out_grad.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::sum_grad(
      x_res, out_grad_res, axis.GetData(), keepdim, reduce_all);
  return Tensor(std::make_shared<LazyTensor>(op_res));
}

template <>
std::vector<Tensor> concat_grad<DescTensor>(const std::vector<Tensor>& x,
                                            const Tensor& out_grad,
                                            const Tensor& axis) {
  std::vector<ir::OpResult> x_res;
  for (uint64_t idx = 0; idx < x.size(); idx++) {
    x_res.emplace_back(std::static_pointer_cast<DescTensor>(x[idx].impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>());
  }

  ir::OpResult out_grad_res =
      std::static_pointer_cast<DescTensor>(out_grad.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>();

  ir::OpResult axis_res = std::static_pointer_cast<DescTensor>(axis.impl())
                              ->getValue()
                              .dyn_cast<ir::OpResult>();

  std::vector<ir::OpResult> op_res =
      paddle::dialect::concat_grad(x_res, out_grad_res, axis_res);

  std::vector<Tensor> op_result;
  for (uint64_t idx = 0; idx < op_res.size(); idx++) {
    op_result.emplace_back(
        std::make_shared<primitive::experimental::DescTensor>(op_res[idx]));
  }
  return op_result;
}

}  // namespace backend
}  // namespace primitive
}  // namespace paddle
