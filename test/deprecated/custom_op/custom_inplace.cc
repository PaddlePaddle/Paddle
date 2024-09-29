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

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

template <typename data_t>
void add_data_pointer(const data_t* x_data, data_t* out_data, int64_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    out_data[i] += x_data[i];
  }
}

template <typename data_t>
void assign_data_pointer(const data_t* x_data,
                         data_t* out_data,
                         int64_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    out_data[i] = x_data[i];
  }
}

template <typename data_t>
void relu_forward_kernel(data_t* x_data, int64_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    x_data[i] = x_data[i] > 0 ? x_data[i] : 0;
  }
}

template <typename data_t>
void relu_backward_kernel(const data_t* out_data,
                          data_t* grad_out_data,
                          int64_t out_numel) {
  for (int64_t i = 0; i < out_numel; ++i) {
    grad_out_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<data_t>(0) ? 1. : 0.);
  }
}

void AddForward(paddle::Tensor& x, const paddle::Tensor& y) {  // NOLINT
  CHECK_INPUT(x);

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "AddForward", ([&] {
        add_data_pointer<data_t>(y.data<data_t>(), x.data<data_t>(), x.size());
      }));
}

std::vector<paddle::Tensor> AddBackward(const paddle::Tensor& x,
                                        const paddle::Tensor& y,
                                        paddle::Tensor& out_grad) {  // NOLINT
  CHECK_INPUT(x);
  CHECK_INPUT(y);

  paddle::Tensor y_grad = paddle::empty(x.shape(), x.dtype(), x.place());

  PD_DISPATCH_FLOATING_TYPES(
      out_grad.type(), "AddBackward", ([&] {
        assign_data_pointer<data_t>(
            out_grad.data<data_t>(), y_grad.data<data_t>(), out_grad.size());
      }));

  return {y_grad};
}

PD_BUILD_OP(custom_add)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetInplaceMap({{"X", "Out"}})
    .SetKernelFn(PD_KERNEL(AddForward));

PD_BUILD_GRAD_OP(custom_add)
    .Inputs({"X", "Y", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X"), paddle::Grad("Y")})
    .SetInplaceMap({{paddle::Grad("Out"), paddle::Grad("X")}})
    .SetKernelFn(PD_KERNEL(AddBackward));

// out[i] = x[i] + y
void AddVectorForward(std::vector<paddle::Tensor>& x,  // NOLINT
                      const paddle::Tensor& y) {
  CHECK_INPUT(y);

  PD_DISPATCH_FLOATING_TYPES(y.type(), "AddVectorForward", ([&] {
                               for (size_t i = 0; i < x.size(); ++i) {
                                 add_data_pointer<data_t>(y.data<data_t>(),
                                                          x[i].data<data_t>(),
                                                          y.size());
                               }
                             }));
}

// dout[i] / dx[i] = out_grad[i] (do not need any code, inplace automatically)
// dout / dy = out_grad[0] + ... + out_grad[n - 1]
std::vector<paddle::Tensor> AddVectorBackward(
    const std::vector<paddle::Tensor>& x,
    const paddle::Tensor& y,
    std::vector<paddle::Tensor>& out_grad) {  // NOLINT
  CHECK_INPUT(x[0]);
  CHECK_INPUT(y);
  PD_CHECK(x.size() == out_grad.size(),
           "x must have the same size as out_grad.");

  paddle::Tensor y_grad = paddle::zeros(y.shape(), y.dtype(), y.place());

  PD_DISPATCH_FLOATING_TYPES(
      y.type(), "AddVectorBackward", ([&] {
        // y_grad = out_grad[0] + ... + out_grad[n - 1]
        for (size_t i = 0; i < out_grad.size(); ++i) {
          add_data_pointer<data_t>(
              out_grad[i].data<data_t>(), y_grad.data<data_t>(), y_grad.size());
        }
      }));
  return {y_grad};
}

PD_BUILD_OP(custom_add_vec)
    .Inputs({paddle::Vec("X"), "Y"})
    .Outputs({paddle::Vec("Out")})
    .SetInplaceMap({{paddle::Vec("X"), paddle::Vec("Out")}})
    .SetKernelFn(PD_KERNEL(AddVectorForward));

PD_BUILD_GRAD_OP(custom_add_vec)
    .Inputs({paddle::Vec("X"), "Y", paddle::Grad(paddle::Vec("Out"))})
    .Outputs({paddle::Grad(paddle::Vec("X")), paddle::Grad("Y")})
    .SetInplaceMap({{paddle::Grad(paddle::Vec("Out")),
                     paddle::Grad(paddle::Vec("X"))}})
    .SetKernelFn(PD_KERNEL(AddVectorBackward));

void MultiInplaceForward(paddle::Tensor& x,  // NOLINT
                         const paddle::Tensor& y,
                         paddle::Tensor& a,  // NOLINT
                         const paddle::Tensor& b) {
  CHECK_INPUT(x);
  CHECK_INPUT(a);

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "MultiInplaceForward", ([&] {
        add_data_pointer<data_t>(y.data<data_t>(), x.data<data_t>(), x.size());
        add_data_pointer<data_t>(b.data<data_t>(), a.data<data_t>(), a.size());
      }));
}

std::vector<paddle::Tensor> MultiInplaceBackward(
    const paddle::Tensor& x,
    const paddle::Tensor& y,
    paddle::Tensor& outxy_grad,  // NOLINT
    const paddle::Tensor& a,
    const paddle::Tensor& b,
    paddle::Tensor& outab_grad) {  // NOLINT
  CHECK_INPUT(x);
  CHECK_INPUT(y);
  CHECK_INPUT(a);
  CHECK_INPUT(b);

  paddle::Tensor y_grad = paddle::empty(x.shape(), x.dtype(), x.place());
  paddle::Tensor b_grad = paddle::empty(a.shape(), a.dtype(), a.place());

  PD_DISPATCH_FLOATING_TYPES(
      outxy_grad.type(), "MultiInplaceBackward", ([&] {
        assign_data_pointer<data_t>(outxy_grad.data<data_t>(),
                                    y_grad.data<data_t>(),
                                    outxy_grad.size());
        assign_data_pointer<data_t>(outab_grad.data<data_t>(),
                                    b_grad.data<data_t>(),
                                    outab_grad.size());
      }));

  return {y_grad, b_grad};
}

PD_BUILD_OP(custom_multi_inplace)
    .Inputs({"X", "Y", "A", "B"})
    .Outputs({"OutXY", "OutAB"})
    .SetInplaceMap({{"X", "OutXY"}, {"A", "OutAB"}})
    .SetKernelFn(PD_KERNEL(MultiInplaceForward));

PD_BUILD_GRAD_OP(custom_multi_inplace)
    .Inputs({"X", "Y", paddle::Grad("OutXY"), "A", "B", paddle::Grad("OutAB")})
    .Outputs({paddle::Grad("X"),
              paddle::Grad("Y"),
              paddle::Grad("A"),
              paddle::Grad("B")})
    .SetInplaceMap({{paddle::Grad("OutXY"), paddle::Grad("X")},
                    {paddle::Grad("OutAB"), paddle::Grad("A")}})
    .SetKernelFn(PD_KERNEL(MultiInplaceBackward));

void ReluForwardInplace(paddle::Tensor& x) {  // NOLINT
  CHECK_INPUT(x);

  PD_DISPATCH_FLOATING_TYPES(x.type(), "ReluForward", ([&] {
                               relu_forward_kernel<data_t>(x.data<data_t>(),
                                                           x.size());
                             }));
}

void ReluBackwardInplace(const paddle::Tensor& x,
                         const paddle::Tensor& out,
                         paddle::Tensor& grad_out) {  // NOLINT
  CHECK_INPUT(out);

  PD_DISPATCH_FLOATING_TYPES(
      grad_out.type(), "ReluBackward", ([&] {
        relu_backward_kernel<data_t>(
            out.data<data_t>(), grad_out.data<data_t>(), grad_out.size());
      }));
}

PD_BUILD_OP(custom_relu_inplace)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetInplaceMap({{"X", "Out"}})
    .SetKernelFn(PD_KERNEL(ReluForwardInplace));

PD_BUILD_GRAD_OP(custom_relu_inplace)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetInplaceMap({{paddle::Grad("Out"), paddle::Grad("X")}})
    .SetKernelFn(PD_KERNEL(ReluBackwardInplace));
