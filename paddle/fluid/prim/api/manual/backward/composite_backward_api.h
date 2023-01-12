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
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace prim {
using Tensor = paddle::experimental::Tensor;
using IntArray =
    paddle::experimental::IntArrayBase<paddle::experimental::Tensor>;
// using IntArray = paddle::experimental::IntArray;
//  This function should have as same signature as phi, which defined in
//  paddle/phi/api/backward/backward_api.h
template <typename T>
void tanh_grad(const Tensor& out, const Tensor& grad_out, Tensor* grad_x) {
  auto tmp = pow<T>(out, 2.0);
  tmp = scale<T>(tmp, -1.0, 1.0, true);
  auto grad_x_tmp = multiply<T>(grad_out, tmp);
  grad_x->set_impl(grad_x_tmp.impl());
}

template <typename T>
void sum_grad(const Tensor& x,
              const Tensor& out_grad,
              const IntArray& axis,
              bool keepdim,
              bool reduce_all,
              Tensor* x_grad) {
  std::vector<int> x_dim = phi::vectorize<int>(x.dims());
  std::cout << "*****x_dim = " << x_dim.size() << std::endl;
  std::cout << " ******out_grad.shape : "
            << phi::vectorize<int>(out_grad.dims()).size() << std::endl;
  int64_t axis_size = axis.size();
  int64_t x_dim_size = x_dim.size();
  reduce_all = false;
  if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
    reduce_all = true;
  } else {
    reduce_all = false;
  }
  auto x_grad_tmp = Tensor();
  if (!keepdim) {
    auto axis_ = std::vector<int64_t>();
    if (reduce_all) {
      for (int64_t i = 1; i < x_dim_size; i++) {
        axis_.push_back(i);
      }
    } else {
      axis_ = axis.GetData();
    }
    std::cout << "******axis_.shape : " << axis_.size() << std::endl;
    auto out_grad_ = unsqueeze<T>(out_grad, axis_);
    std::cout << "*******out_grad_.shape : "
              << phi::vectorize<int>(out_grad_.dims()).size() << std::endl;
    x_grad_tmp = expand<T>(out_grad_, x_dim);
  } else {
    x_grad_tmp = expand<T>(out_grad, x_dim);
  }
  std::cout << "*******x_grad_tmp.shape : "
            << phi::vectorize<int>(x_grad_tmp.dims()).size() << std::endl;

  x_grad->set_impl(x_grad_tmp.impl());
}

}  // namespace prim
}  // namespace paddle
