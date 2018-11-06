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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/conv_transpose_op.h"
#include "paddle/fluid/platform/assert.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;

static constexpr size_t kConvCUDNNWorkspaceLimitBytes = 1024 * 1024 * 1024;

template <typename T>
class CUDNNConvTransposeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    // cudnn v5 does not support dilations
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    int user_workspace_size = ctx.Attr<int>("workspace_size_MB");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedFilterDescriptor filter_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout;

    if (strides.size() == 2U) {
      layout = DataLayout::kNCHW;
    } else {
      layout = DataLayout::kNCDHW;
    }

    // (N, M, H, W) or (N, M, D, H, W)
    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()), groups);
    // (N, C, O_h, O_w) or (N, C, O_d, O_h, O_w)
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output->dims()), groups);
    // (M, C, K_h, K_w) or (M, C, K_d, K_h, K_w)
    cudnnFilterDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()), groups);
    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

    // ------------------- cudnn conv workspace ---------------------
    size_t workspace_size_in_bytes;  // final workspace to allocate.
    size_t workspace_size_limit = kConvCUDNNWorkspaceLimitBytes;
    if (user_workspace_size > 0) {
      workspace_size_limit = user_workspace_size * 1024 * 1024;
    }
    // ------------------- cudnn conv algorithm ---------------------
    cudnnConvolutionBwdDataAlgo_t algo;
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    // Get the algorithm
    CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
        handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
        // dxDesc: Handle to the previously initialized output tensor
        // descriptor.
        cudnn_output_desc, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_size_limit, &algo));

    // get workspace size able to allocate
    CUDNN_ENFORCE(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
            cudnn_output_desc, algo, &workspace_size_in_bytes));

    // ------------------- cudnn conv transpose forward ---------------------
    int input_offset = input->numel() / input->dims()[0] / groups;
    int output_offset = output->numel() / output->dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    T alpha = 1.0f, beta = 0.0f;
    for (int g = 0; g < groups; g++) {
      auto cudnn_func = [&](void* cudnn_workspace) {
        CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
            handle, &alpha, cudnn_filter_desc, filter_data + filter_offset * g,
            cudnn_input_desc, input_data + input_offset * g, cudnn_conv_desc,
            algo, cudnn_workspace, workspace_size_in_bytes, &beta,
            cudnn_output_desc, output_data + output_offset * g));
      };
      dev_ctx.RunCudnnFuncWithWorkspace(cudnn_func, workspace_size_in_bytes);
    }
  }
};

template <typename T>
class CUDNNConvTransposeGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto input = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));
    const T* input_data = input->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    const T* filter_data = filter->data<T>();

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    // cudnn v5 does not support dilations
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    int user_workspace_size = ctx.Attr<int>("workspace_size_MB");

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedFilterDescriptor filter_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout = DataLayout::kNCHW;

    // Input: (N, M, H, W) or (N, M, D, H, W)
    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()), groups);
    // Output: (N, C, O_h, O_w) or (N, C, O_d, O_h, O_w)
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output_grad->dims()), groups);
    // Filter (M, C, K_h, K_w) or (M, C, K_d K_h, K_w)
    cudnnFilterDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()), groups);

    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

    // ------------------- cudnn backward algorithm ---------------------
    cudnnConvolutionFwdAlgo_t data_algo;
    cudnnConvolutionBwdFilterAlgo_t filter_algo;
    size_t bwd_filter_ws_size, fwd_ws_size;
    size_t workspace_size_in_bytes = 0;
    size_t workspace_size_limit = kConvCUDNNWorkspaceLimitBytes;
    if (user_workspace_size > 0) {
      workspace_size_limit = user_workspace_size * 1024 * 1024;
    }

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    if (input_grad) {
      // choose backward algorithm for data
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm(
          handle, cudnn_output_desc, cudnn_filter_desc, cudnn_conv_desc,
          cudnn_input_desc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          workspace_size_limit, &data_algo));
      CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
          handle, cudnn_output_desc, cudnn_filter_desc, cudnn_conv_desc,
          cudnn_input_desc, data_algo, &fwd_ws_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, fwd_ws_size);
    }

    if (filter_grad) {
      // choose backward algorithm for filter
      CUDNN_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              handle, cudnn_output_desc, cudnn_input_desc, cudnn_conv_desc,
              cudnn_filter_desc,
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &filter_algo));

      // get workspace for backwards filter algorithm
      CUDNN_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
              handle, cudnn_output_desc, cudnn_input_desc, cudnn_conv_desc,
              cudnn_filter_desc, filter_algo, &bwd_filter_ws_size));
      workspace_size_in_bytes =
          std::max(workspace_size_in_bytes, bwd_filter_ws_size);
    }

    // ------------------- cudnn conv backward data ---------------------
    // FIXME(typhoonzero): template type T may not be the same as cudnn call.
    int input_offset = input->numel() / input->dims()[0] / groups;
    int output_grad_offset =
        output_grad->numel() / output_grad->dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    T alpha = 1.0f, beta = 0.0f;
    if (input_grad) {
      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset input_grad.
      for (int g = 0; g < groups; g++) {
        auto cudnn_func = [&](void* cudnn_workspace) {
          CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
              handle, &alpha, cudnn_output_desc,
              output_grad_data + output_grad_offset * g, cudnn_filter_desc,
              filter_data + filter_offset * g, cudnn_conv_desc, data_algo,
              cudnn_workspace, workspace_size_in_bytes, &beta, cudnn_input_desc,
              input_grad_data + input_offset * g));
        };
        dev_ctx.RunCudnnFuncWithWorkspace(cudnn_func, workspace_size_in_bytes);
      }
    }

    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      T* filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset filter_grad.
      // Gradient with respect to the filter
      for (int g = 0; g < groups; g++) {
        auto cudnn_func = [&](void* cudnn_workspace) {
          CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
              handle, &alpha, cudnn_output_desc,
              output_grad_data + output_grad_offset * g, cudnn_input_desc,
              input_data + input_offset * g, cudnn_conv_desc, filter_algo,
              cudnn_workspace, workspace_size_in_bytes, &beta,
              cudnn_filter_desc, filter_grad_data + filter_offset * g));
        };
        dev_ctx.RunCudnnFuncWithWorkspace(cudnn_func, workspace_size_in_bytes);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(conv2d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<float>,
                   ops::CUDNNConvTransposeOpKernel<double>);
REGISTER_OP_KERNEL(conv2d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<float>,
                   ops::CUDNNConvTransposeGradOpKernel<double>);

REGISTER_OP_KERNEL(conv3d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<float>,
                   ops::CUDNNConvTransposeOpKernel<double>);
REGISTER_OP_KERNEL(conv3d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<float>,
                   ops::CUDNNConvTransposeGradOpKernel<double>);
