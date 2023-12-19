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

#include "paddle/extension.h"

std::vector<paddle::Tensor> relu_cuda_forward(const paddle::Tensor& x);
std::vector<paddle::Tensor> relu_cuda_backward(const paddle::Tensor& x,
                                               const paddle::Tensor& out,
                                               const paddle::Tensor& grad_out);

std::vector<paddle::Tensor> ReluForward(const paddle::Tensor& x);

std::vector<paddle::Tensor> ReluBackward(const paddle::Tensor& x,
                                         const paddle::Tensor& out,
                                         const paddle::Tensor& grad_out);

// Reuse codes in `custom_relu_op.cc/cu` to register another custom operator
// to test jointly compile multi operators at same time.
PD_BUILD_OP(custom_relu_dup)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ReluForward));

PD_BUILD_GRAD_OP(custom_relu_dup)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(ReluBackward));
