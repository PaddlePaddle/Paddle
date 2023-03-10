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
// WIdata_tHOUdata_t WARRANdata_tIES OR CONDIdata_tIONS OF ANY KIND, either
// express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>

#include "paddle/extension.h"

template <typename data_t>
void add_forward_kernel(data_t* x_data, const data_t* y_data, int64_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    x_data[i] += y_data[i];
  }
}

template <typename data_t>
void add_backward_kernel(data_t* y_grad_data,
                         const data_t* out_grad_data,
                         int64_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    y_grad_data[i] = out_grad_data[i];
  }
}

void AddForward(paddle::Tensor* x, const paddle::Tensor& y) {
  PD_CHECK(x->place() == paddle::PlaceType::kCPU, "x must be a CPU Tensor.");

  PD_DISPATCH_FLOATING_TYPES(x->type(), "AddForward", ([&] {
                               add_forward_kernel<data_t>(x->data<data_t>(),
                                                          y.data<data_t>(),
                                                          x->size());
                             }));
}

std::vector<paddle::Tensor> AddBackward(const paddle::Tensor& x,
                                        const paddle::Tensor& y,
                                        paddle::Tensor* out_grad) {
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, "x must be a CPU Tensor.");

  paddle::Tensor y_grad = paddle::empty(x.shape(), x.dtype(), x.place());

  PD_DISPATCH_FLOATING_TYPES(
      out_grad->type(), "AddBackward", ([&] {
        add_backward_kernel<data_t>(
            y_grad.data<data_t>(), out_grad->data<data_t>(), out_grad->size());
      }));

  return {y_grad};
}

PD_BUILD_OP(custom_add)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .Inplace({{"X", "Out"}})
    .SetKernelFn(PD_KERNEL(AddForward));

PD_BUILD_GRAD_OP(custom_add)
    .Inputs({"X", "Y", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X"), paddle::Grad("Y")})
    .Inplace({{paddle::Grad("Out"), paddle::Grad("X")}})
    .SetKernelFn(PD_KERNEL(AddBackward));

// 采取什么注册形式？
// PD_BUILD_OP(custom_add)
//     .Inputs({"X", "Y"})
//     .Outputs({"Out"})
//     .SetKernelFn(PD_KERNEL(AddForward))
//     .Inplace({"X": "Out"});

// 一种两难的境地：
// 1. 将输出全部以指针的方式放到函数里，面对多个输出的情况，无法处理
// 2. 将输出全部放到返回值里，没办法处理 inplace，输出设置为 vector<Tensor *>
// 或者 vector<Tensor> 都不可以，因为总有 Tensor 需要指针或者需要 Tensor 做拷贝
// custom_add(Tensor* x, const Tensor& y)

// PD_BUILD_OP(custom_add)
//     .Inputs({"X", "Y"})
//     .Outputs({Inplace("X")})
//     .SetKernelFn(PD_KERNEL(AddForward))

// (out_grad -> x_grad)
// vector<Tensor> custom_add(const Tensor& x, const Tensor& y, Tensor* out_grad)
// out_grad: inplace, y_grad: vector<Tensor>

// PD_BUILD_GRAD_OP(custom_add)
//     .Inputs({"X", "Y", paddle::Grad(Inplace("X"))})
//     .Outputs({Inplace(paddle::Grad(Inplace("X"))), paddle::Grad("Y")})
//     .SetKernelFn(PD_KERNEL(AddBackward))
