/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the spopecific language governing permissions and
limitations under the License. */

#include <utility>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/memory.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/operators/conv_miopen_helper.h"
#else
#include "paddle/fluid/operators/conv_cudnn_helper.h"
#endif
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/operators/math/padding.h"
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(cudnn_deterministic);
DECLARE_uint64(conv_workspace_size_limit);
DECLARE_bool(cudnn_exhaustive_search);

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;

static inline bool IsVoltaOrLater(const platform::CUDADeviceContext& dev_ctx) {
  return dev_ctx.GetComputeCapability() >= 70;
}

template <typename T>
class CUDNNConvOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

template <typename T>
class CUDNNConvGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

/*
 * Inputs:  I, W, dO, ddI, ddW
 * Outputs: ddO, dW, dI
 * ddo = conv(ddI, W) + conv(I, ddW)
 * dW = conv_bp_filter(ddI, dO)
 * dI = conv_bp_data(ddW, dO)
 */
template <typename T>
class CUDNNConvDoubleGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_KERNEL(conv2d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv2d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::float16>);
REGISTER_OP_KERNEL(
    conv2d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);
// ROCM has limit thread in depthwise_conv.cu and willl result in accuracy issue
// Use depthwise_conv2d in MIOPEN to resolve this issue
REGISTER_OP_KERNEL(depthwise_conv2d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(depthwise_conv2d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    depthwise_conv2d_grad_grad,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);

REGISTER_OP_KERNEL(conv3d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv3d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>);
REGISTER_OP_KERNEL(
    conv3d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
REGISTER_OP_KERNEL(conv2d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>,
                   paddle::operators::CUDNNConvOpKernel<plat::bfloat16>);
REGISTER_OP_KERNEL(conv2d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::float16>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::bfloat16>);
REGISTER_OP_KERNEL(
    conv2d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::bfloat16>);

REGISTER_OP_CUDA_KERNEL(
    depthwise_conv2d_grad_grad,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::bfloat16>);
#else
REGISTER_OP_KERNEL(conv2d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv2d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>,
                   paddle::operators::CUDNNConvGradOpKernel<plat::float16>);
REGISTER_OP_KERNEL(
    conv2d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    depthwise_conv2d_grad_grad,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);
#endif

REGISTER_OP_KERNEL(conv3d, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvOpKernel<float>,
                   paddle::operators::CUDNNConvOpKernel<double>,
                   paddle::operators::CUDNNConvOpKernel<plat::float16>);
REGISTER_OP_KERNEL(conv3d_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNConvGradOpKernel<float>,
                   paddle::operators::CUDNNConvGradOpKernel<double>);
REGISTER_OP_KERNEL(
    conv3d_grad_grad, CUDNN, plat::CUDAPlace,
    paddle::operators::CUDNNConvDoubleGradOpKernel<float>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<double>,
    paddle::operators::CUDNNConvDoubleGradOpKernel<plat::float16>);
#endif
