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
      x_res, out_grad_res, axis.GetData(), keepdim, reduce_all);

  return Tensor(std::make_shared<primitive::experimental::DescTensor>(op_res));
}

}  // namespace experimental
}  // namespace backend
}  // namespace primitive
}  // namespace paddle
