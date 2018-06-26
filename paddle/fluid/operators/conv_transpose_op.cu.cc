/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/conv_transpose_op.h"

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;

// conv2d
REGISTER_OP_CUDA_KERNEL(conv2d_transpose,
                        ops::GemmConvTransposeKernel<CUDA, float>,
                        ops::GemmConvTransposeKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(conv2d_transpose_grad,
                        ops::GemmConvTransposeGradKernel<CUDA, float>,
                        ops::GemmConvTransposeGradKernel<CUDA, double>);

// conv3d
REGISTER_OP_CUDA_KERNEL(conv3d_transpose,
                        ops::GemmConvTransposeKernel<CUDA, float>,
                        ops::GemmConvTransposeKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(conv3d_transpose_grad,
                        ops::GemmConvTransposeGradKernel<CUDA, float>,
                        ops::GemmConvTransposeGradKernel<CUDA, double>);

// depthwise conv2d
REGISTER_OP_CUDA_KERNEL(depthwise_conv2d_transpose,
                        ops::DepthwiseConvTransposeKernel<CUDA, float>,
                        ops::DepthwiseConvTransposeKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(depthwise_conv2d_transpose_grad,
                        ops::DepthwiseConvTransposeGradKernel<CUDA, float>,
                        ops::DepthwiseConvTransposeGradKernel<CUDA, double>);
