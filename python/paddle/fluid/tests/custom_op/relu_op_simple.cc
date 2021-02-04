// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <vector>

#include "paddle/extension.h"

template <typename data_t>
void relu_cpu_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             int64_t x_numel) {
  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = std::max(static_cast<data_t>(0.), x_data[i]);
  }
}

template <typename data_t>
void relu_cpu_backward_kernel(const data_t* grad_out_data,
                              const data_t* out_data,
                              data_t* grad_x_data,
                              int64_t out_numel) {
  for (int i = 0; i < out_numel; ++i) {
    grad_x_data[i] =
        grad_out_data[i] * (out_data[i] > static_cast<data_t>(0) ? 1. : 0.);
  }
}

std::vector<paddle::CustomTensor> relu_cpu_forward(const paddle::CustomTensor& x) {
  auto out = paddle::CustomTensor(paddle::PaddlePlace(paddle::PlaceType::kCPU));
  out.Reshape(x.shape());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "relu_cpu_forward", ([&] {
        relu_cpu_forward_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(x.place()), x.size());
      }));

  return {out};
}

std::vector<paddle::CustomTensor> relu_cpu_backward(const paddle::CustomTensor& grad_out,
                                              const paddle::CustomTensor& out,
                                              const paddle::CustomTensor& x) {
  auto grad_x = paddle::CustomTensor(paddle::PaddlePlace(paddle::PlaceType::kCPU));
  grad_x.Reshape(x.shape());

  PD_DISPATCH_FLOATING_TYPES(out.type(), "relu_cpu_backward", ([&] {
                               relu_cpu_backward_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   out.data<data_t>(),
                                   grad_x.mutable_data<data_t>(x.place()),
                                   out.size());
                             }));

  return {grad_x};
}

std::vector<paddle::CustomTensor> relu_cuda_forward(const paddle::CustomTensor& x);
std::vector<paddle::CustomTensor> relu_cuda_backward(const paddle::CustomTensor& grad_out,
                                               const paddle::CustomTensor& out,
                                               const paddle::CustomTensor& x);

std::vector<paddle::CustomTensor> ReluForward(const paddle::CustomTensor& x) {
  // TODO(chenweihang): Check Input
    if (x.place().GetPlace() == paddle::PlaceType::kCPU) {
        return relu_cpu_forward(x);
    } else if (x.place().GetPlace() == paddle::PlaceType::kGPU) {
        return relu_cuda_forward(x);
    } else {
        throw std::runtime_error("Not implemented.");
    }
}

std::vector<paddle::CustomTensor> ReluBackward(const paddle::CustomTensor& grad_out,
                                         const paddle::CustomTensor& out,
                                         const paddle::CustomTensor& x) {
  // TODO(chenweihang): Check Input
    if (x.place().GetPlace() == paddle::PlaceType::kCPU) {
        return relu_cpu_backward(grad_out, out, x);
    } else if (x.place().GetPlace() == paddle::PlaceType::kGPU) {
        return relu_cuda_backward(grad_out, out, x);
    } else {
        throw std::runtime_error("Not implemented.");
    }
}

std::vector<std::vector<int64_t>> ReluInferShape(std::vector<int64_t> x_shape) {
  return {x_shape};
}

BUILD_OPERATOR(relu2,
               OP_INFO(ReluForward),
               PD_KERNEL(ReluForward),
               PD_KERNEL(ReluBackward),
               PD_INFER_SHAPE(ReluInferShape));
