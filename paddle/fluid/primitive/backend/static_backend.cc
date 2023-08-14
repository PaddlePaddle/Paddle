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
#include "paddle/fluid/primitive/type/desc_tensor.h"

namespace paddle {
namespace primitive {
namespace backend {
namespace experimental {

using DescTensor = paddle::primitive::experimental::DescTensor;

template <>
Tensor tanh_grad<DescTensor>(const Tensor& out, const Tensor& grad_out) {
  ir::OpResult out_res = std::static_pointer_cast<DescTensor>(out.impl())
                             ->getValue()
                             .dyn_cast<ir::OpResult>();
  ir::OpResult grad_out_res =
      std::static_pointer_cast<DescTensor>(grad_out.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>();

  ir::OpResult op_res = paddle::dialect::tanh_grad(out_res, grad_out_res);

  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor mean_grad<DescTensor>(const Tensor& x,
                             const Tensor& out_grad,
                             const IntArray& axis,
                             bool keepdim,
                             bool reduce_all) {
  ir::OpResult x_res = std::static_pointer_cast<DescTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult out_grad_res =
      std::static_pointer_cast<DescTensor>(out_grad.impl())
          ->getValue()
          .dyn_cast<ir::OpResult>();

  ir::OpResult op_res = paddle::dialect::mean_grad(
      x_res, out_grad_res, axis, keepdim, reduce_all);

  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor divide<DescTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<DescTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<DescTensor>(y.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::divide(x_res, y_res);
  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor add<DescTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<DescTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<DescTensor>(y.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::add(x_res, y_res);
  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor multiply<DescTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<DescTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<DescTensor>(y.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::multiply(x_res, y_res);
  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor elementwise_pow<DescTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<DescTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<DescTensor>(y.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::elementwise_pow(x_res, y_res);
  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor scale<DescTensor>(const Tensor& x,
                         const Scalar& scale,
                         float bias,
                         bool bias_after_scale) {
  ir::OpResult x_res = std::static_pointer_cast<DescTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res =
      paddle::dialect::scale(x_res, scale.to<float>(), bias, bias_after_scale);
  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor sum<DescTensor>(const Tensor& x,
                       const IntArray& axis,
                       phi::DataType dtype,
                       bool keepdim) {
  ir::OpResult x_res = std::static_pointer_cast<DescTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res =
      paddle::dialect::sum(x_res, axis.GetData(), dtype, keepdim);
  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor full<DescTensor>(const IntArray& shape,
                        const Scalar& value,
                        phi::DataType dtype,
                        phi::Place place) {
  ir::OpResult op_res =
      paddle::dialect::full(shape.GetData(), value.to<float>(), dtype, place);
  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor reshape<DescTensor>(const Tensor& x, const IntArray& shape) {
  ir::OpResult x_res = std::static_pointer_cast<DescTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::reshape(x_res, shape.GetData());
  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor expand<DescTensor>(const Tensor& x, const IntArray& shape) {
  ir::OpResult x_res = std::static_pointer_cast<DescTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::expand(x_res, shape.GetData());
  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

template <>
Tensor tile<DescTensor>(const Tensor& x, const IntArray& repeat_times) {
  ir::OpResult x_res = std::static_pointer_cast<DescTensor>(x.impl())
                           ->getValue()
                           .dyn_cast<ir::OpResult>();
  ir::OpResult op_res = paddle::dialect::tile(x_res, repeat_times.GetData());
  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

}  // namespace experimental
}  // namespace backend
}  // namespace primitive
}  // namespace paddle
