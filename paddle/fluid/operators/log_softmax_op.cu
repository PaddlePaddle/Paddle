// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/log_softmax_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class LogSoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    int input_axis = ctx.Attr<int>("axis");
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    phi::SoftmaxForwardCUDAKernelDriver<T, true>(dev_ctx, *x, input_axis, out);
  }
};

template <typename T>
class LogSoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    int input_axis = ctx.Attr<int>("axis");
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    phi::SoftmaxBackwardCUDAKernelDriver<T, true>(dev_ctx, *out, *dout,
                                                  input_axis, dx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    log_softmax, ops::LogSoftmaxKernel<plat::CUDADeviceContext, float>,
    ops::LogSoftmaxKernel<plat::CUDADeviceContext, double>,
    ops::LogSoftmaxKernel<plat::CUDADeviceContext, plat::float16>,
    ops::LogSoftmaxKernel<plat::CUDADeviceContext, plat::bfloat16>);
REGISTER_OP_CUDA_KERNEL(
    log_softmax_grad, ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, float>,
    ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, double>,
    ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, plat::bfloat16>);
#ifdef PADDLE_WITH_HIP
REGISTER_OP_KERNEL(log_softmax, CUDNN, plat::CUDAPlace,
                   ops::LogSoftmaxCUDNNKernel<float>,
                   ops::LogSoftmaxCUDNNKernel<plat::float16>,
                   ops::LogSoftmaxCUDNNKernel<plat::bfloat16>);
REGISTER_OP_KERNEL(log_softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::LogSoftmaxGradCUDNNKernel<float>,
                   ops::LogSoftmaxGradCUDNNKernel<plat::float16>,
                   ops::LogSoftmaxGradCUDNNKernel<plat::bfloat16>);
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
REGISTER_OP_KERNEL(log_softmax, CUDNN, plat::CUDAPlace,
                   ops::LogSoftmaxCUDNNKernel<float>,
                   ops::LogSoftmaxCUDNNKernel<double>,
                   ops::LogSoftmaxCUDNNKernel<plat::float16>,
                   ops::LogSoftmaxCUDNNKernel<plat::bfloat16>);
REGISTER_OP_KERNEL(log_softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::LogSoftmaxGradCUDNNKernel<float>,
                   ops::LogSoftmaxGradCUDNNKernel<double>,
                   ops::LogSoftmaxGradCUDNNKernel<plat::float16>,
                   ops::LogSoftmaxGradCUDNNKernel<plat::bfloat16>);
#else
REGISTER_OP_KERNEL(log_softmax, CUDNN, plat::CUDAPlace,
                   ops::LogSoftmaxCUDNNKernel<float>,
                   ops::LogSoftmaxCUDNNKernel<double>,
                   ops::LogSoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(log_softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::LogSoftmaxGradCUDNNKernel<float>,
                   ops::LogSoftmaxGradCUDNNKernel<double>,
                   ops::LogSoftmaxGradCUDNNKernel<plat::float16>);
#endif
#endif
