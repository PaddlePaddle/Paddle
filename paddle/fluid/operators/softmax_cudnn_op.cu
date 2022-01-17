/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/softmax_cudnn_op.cu.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(log_softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float, true>,
                   ops::SoftmaxCUDNNKernel<plat::float16, true>);
REGISTER_OP_KERNEL(log_softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float, true>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16, true>);
#else
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<double>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<double>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(log_softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float, true>,
                   ops::SoftmaxCUDNNKernel<double, true>,
                   ops::SoftmaxCUDNNKernel<plat::float16, true>);
REGISTER_OP_KERNEL(log_softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float, true>,
                   ops::SoftmaxGradCUDNNKernel<double, true>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16, true>);
#endif
