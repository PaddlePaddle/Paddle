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
#include "paddle/fluid/prim/api/generated/prim_api/prim_api.h"
#include "paddle/fluid/prim/api/manual/utils/utils.h"
namespace paddle {
namespace prim {

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
void subtract_grad(const Tensor& x,
                   const Tensor& y,
                   const Tensor& out_grad,
                   int axis,
                   Tensor* dx,
                   Tensor* dy) {
  if (dy) {
    auto scale_out_grad = scale<T>(out_grad, -1.0, 0.0, true);
    if (phi::product(x.dims()) > phi::product(y.dims())) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(x.dims(), y.dims());
      auto dy_reduce_res =
          sum<T>(scale_out_grad, phi::vectorize(reduce_dim), y.dtype(), false);
      auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
      dy->set_impl(dy_tmp.impl());
    } else {
      by_pass<T>(scale_out_grad, dy);
    }
  }
  if (dx) {
    if (phi::product(y.dims()) > phi::product(x.dims())) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(y.dims(), x.dims());
      auto dx_reduce_res =
          sum<T>(out_grad, phi::vectorize(reduce_dim), x.dtype(), false);
      auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
      dx->set_impl(dx_tmp.impl());
    } else {
      by_pass<T>(out_grad, dx);
    }
  }
}

template <typename T>
void add_grad(const Tensor& x,
              const Tensor& y,
              const Tensor& out_grad,
              int axis,
              Tensor* dx,
              Tensor* dy) {
  if (dy) {
    if (phi::product(x.dims()) > phi::product(y.dims())) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(x.dims(), y.dims());
      auto dy_reduce_res =
          sum<T>(out_grad, phi::vectorize(reduce_dim), y.dtype(), false);
      auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
      dy->set_impl(dy_tmp.impl());
    } else {
      by_pass<T>(out_grad, dy);
    }
  }
  if (dx) {
    if (phi::product(y.dims()) > phi::product(x.dims())) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(y.dims(), x.dims());
      auto dx_reduce_res =
          sum<T>(out_grad, phi::vectorize(reduce_dim), x.dtype(), false);
      auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
      dx->set_impl(dx_tmp.impl());
    } else {
      by_pass<T>(out_grad, dx);
    }
  }
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
    if (phi::product(x.dims()) > phi::product(y.dims())) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(x.dims(), y.dims());
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
    if (phi::product(y.dims()) > phi::product(x.dims())) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(y.dims(), x.dims());
      auto dx_reduce_res =
          sum<T>(dx_res, phi::vectorize(reduce_dim), x.dtype(), false);
      auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
      dx->set_impl(dx_tmp.impl());
    } else {
      dx->set_impl(dx_res.impl());
    }
  }  // indicate we will compute dx
}
}  // namespace prim
}  // namespace paddle
