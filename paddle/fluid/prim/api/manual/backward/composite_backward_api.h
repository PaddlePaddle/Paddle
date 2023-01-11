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

#pragma once
#include "paddle/fluid/prim/api/manual/prim_api/prim_api.h"
#include "paddle/fluid/prim/api/manual/utils/utils.h"
namespace paddle {
namespace prim {

// template <typename T>
// Tensor reduce_as(const Tensor& x, const Tensor& ref) {
//   auto res = empty<DescTensor>({}, x.dtype, paddle::Place());
//   if (x.dims() != ref.dims()) {
//     res = sum<T>(res, phi::vectorize(get_reduce_dims(x_grad_no_reduce.dims(),
//     x.dims())), x.dtype(), false); if (res.dims().size() !=
//     ref.dims().size()) {
//       res = reshape<T>(res, phi::vectorize(ref.dims()));
//     }
//   }
//   return res;
// }

// This function should have as same signature as phi, which defined in
// paddle/phi/api/backward/backward_api.h
template <typename T>
void tanh_grad(const Tensor& out, const Tensor& grad_out, Tensor* grad_x) {
  auto tmp = pow<T>(out, 2.0);
  tmp = scale<T>(tmp, -1.0, 1.0, true);
  auto grad_x_tmp = multiply<T>(grad_out, tmp);
  grad_x->set_impl(grad_x_tmp.impl());
}

template <typename T>
void divide_grad(const Tensor& x,
                 const Tensor& y,
                 const Tensor& out,
                 const Tensor& out_grad,
                 int axis,
                 Tensor* dx,
                 Tensor* dy) {
  if (dy) {
    // dy = -(x/y^2) * dout
    auto tmp0 = pow<T>(y, 2.0);
    auto tmp1 = divide<T>(x, tmp0);
    auto tmp2 = scale<T>(tmp1, -1.0, 0.0, true);
    auto dy_res = multiply<T>(tmp2, out_grad);
    VLOG(3) << "dy_res dims: " << dy_res.dims();
    if (out_grad.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(dy_res.dims(), y.dims());
      auto dy_reduce_res =
          sum<T>(dy_res, phi::vectorize(reduce_dim), y.dtype(), false);
      auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
      dy->set_impl(dy_tmp.impl());
    } else {
      dy->set_impl(dy_res.impl());
    }
  }  // indicate we will compute dy
  if (dx) {
    // dx = (1/y) * dout
    auto one_tensor = full<T>(phi::vectorize(y.dims()), 1.0);
    auto tmp0 = divide<T>(one_tensor, y);
    auto dx_res = multiply<T>(tmp0, out_grad);
    VLOG(3) << "dx_res dims: " << dx_res.dims();
    if (out_grad.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(dx_res.dims(), x.dims());
      auto dx_reduce_res =
          sum<T>(dx_res, phi::vectorize(reduce_dim), x.dtype(), false);
      auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
      dx->set_impl(dx_tmp.impl());
    } else {
      dx->set_impl(dx_res.impl());
    }
  }  // indicate we will compute dx
}

template <typename T>
void multiply_grad(const Tensor& x,
                   const Tensor& y,
                   const Tensor& out_grad,
                   int axis,
                   Tensor* x_grad,
                   Tensor* y_grad) {
  if (x_grad) {
    auto x_grad_unreduce = multiply<T>(out_grad, y);
    if (x.dims() != x_grad_unreduce.dims()) {
      auto x_grad_reduced = sum<T>(
          res,
          phi::vectorize(get_reduce_dims(x_grad_unreduce.dims(), x.dims())),
          x_grad_unreduce.dtype(),
          false);
      if (x_grad_reduced.dims().size() != x.dims().size()) {
        x_grad_reduced = reshape<T>(x_grad_reduced, x.shape());
      }
      x_grad->set_impl(x_grad_reduced.impl());
    }
  }
  if (y_grad) {
    auto y_grad_unreduce = multiply<T>(out_grad, x);
    if (y.dims() != y_grad_unreduce.dims()) {
      auto y_grad_reduced = sum<T>(
          res,
          phi::vectorize(get_reduce_dims(y_grad_unreduce.dims(), x.dims())),
          y_grad_unreduce.dtype(),
          false);
      if (y_grad_reduced.dims().size() != x.dims().size()) {
        y_grad_reduced = reshape<T>(y_grad_reduced, x.shape());
      }
      y_grad->set_impl(y_grad_reduced.impl());
    }
  }
}
}  // namespace prim
}  // namespace paddle
