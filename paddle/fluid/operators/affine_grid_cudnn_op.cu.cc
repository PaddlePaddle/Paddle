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
    //    const T* size_data = size->data<T>();

    Tensor h_sizes;
    framework::TensorCopy(*size, platform::CPUPlace(), &h_sizes);
    const int* h_size_data = h_sizes.data<int>();

    T* output_data = output->mutable_data<T>(
        {h_size_data[0], h_size_data[2], h_size_data[3], 2}, ctx.GetPlace());
    ScopedSpatialTransformerDescriptor st_desc;
    cudnnSpatialTransformerDescriptor_t cudnn_st_desc =
        st_desc.descriptor<T>(4, h_size_data);

    PADDLE_ENFORCE(platform::dynload::cudnnSpatialTfGridGeneratorForward(
        handle, cudnn_st_desc, theta_data, output_data));
  }
};

template <typename T>
class CUDNNAffineGridGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto* size = ctx.Input<Tensor>("Size");
    //    const T* size_data = size->data<T>();
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto theta_grad = ctx.Output<Tensor>(framework::GradVarName("Theta"));
    Tensor h_sizes;
    framework::TensorCopy(*size, platform::CPUPlace(), &h_sizes);
    const int* h_size_data = h_sizes.data<int>();
    ScopedSpatialTransformerDescriptor st_desc;
    cudnnSpatialTransformerDescriptor_t cudnn_st_desc =
        st_desc.descriptor<T>(4, h_size_data);

    const T* output_grad_data = output_grad->data<T>();
    T* theta_grad_data = theta_grad->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE(platform::dynload::cudnnSpatialTfGridGeneratorBackward(
        handle, cudnn_st_desc, output_grad_data, theta_grad_data));
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
REGISTER_OP_KERNEL(affine_grid, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNAffineGridOpKernel<float>);
REGISTER_OP_KERNEL(affine_grid_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNAffineGridGradOpKernel<float>);
