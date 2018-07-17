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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedSpatialTransformerDescriptor =
    platform::ScopedSpatialTransformerDescriptor;

template <typename T>
class CUDNNAffineGridOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto* theta = ctx.Input<Tensor>("Theta");
    auto* size = ctx.Input<Tensor>("Size");
    auto* output = ctx.Output<Tensor>("Output");
    const T* theta_data = theta->data<T>();
    const T* size_data = size->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    ScopedSpatialTransformerDescriptor st_desc;
    cudnnSpatialTransformerDescriptor_t cudnn_st_desc = st_desc.descriptor<T>(
        SamplerType.CUDNN_SAMPLER_BILINEAR, 4,
        {size_data[0], size_data[1], size_data[2], size_data[3]});

    PADDLE_ENFORCE(platform::dynload::cudnnSpatialTfGridGeneratorForward(
        handle, st_desc, theta_data, output_data));
  }
};

template <typename T>
class CUDNNAffineGridGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto* size = ctx.Input<Tensor>("Size");
    const T* size_data = size->data<T>();
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto theta_grad = ctx.Output<Tensor>(framework::GradVarName("Theta"));
    ScopedSpatialTransformerDescriptor st_desc;
    cudnnSpatialTransformerDescriptor_t cudnn_st_desc = st_desc.descriptor<T>(
        SamplerType.CUDNN_SAMPLER_BILINEAR, 4,
        {size_data[0], size_data[1], size_data[2], size_data[3]});

    const T* output_grad_data = output_grad->data<T>();
    T* theta_grad_data = theta_grad->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE(platform::dynload::cudnnSpatialTfGridGeneratorBackward(
        handle, stDesc, output_grad_data, theta_grad_data));
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
REGISTER_OP_KERNEL(affine_grid, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNAffineGridOpKernel<int>,
                   paddle::operators::CUDNNAffineGridOpKernel<int64_t>);
REGISTER_OP_KERNEL(affine_grid_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNAffineGridGradOpKernel<int>,
                   paddle::operators::CUDNNAffineGridGradOpKernel<int64_t>);
