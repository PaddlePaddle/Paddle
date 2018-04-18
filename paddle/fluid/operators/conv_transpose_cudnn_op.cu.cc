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
#include "paddle/fluid/platform/miopen_helper.h"

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
    miopenTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()));
    // (N, C, O_h, O_w) or (N, C, O_d, O_h, O_w)
    miopenTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output->dims()));
    // (M, C, K_h, K_w) or (M, C, K_d, K_h, K_w)
    miopenTensorDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()));
    miopenConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

    // ------------------- cudnn conv workspace ---------------------
    void* cudnn_workspace = nullptr;
    size_t workspace_size_in_bytes;  // final workspace to allocate.
    //size_t workspace_size_limit = kConvCUDNNWorkspaceLimitBytes;
    //if (user_workspace_size > 0) {
    //  workspace_size_limit = user_workspace_size * 1024 * 1024;
    //}
    // ------------------- cudnn conv algorithm ---------------------
    miopenConvBwdDataAlgorithm_t algo;
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.miopen_handle();
    miopenConvAlgoPerf_t perfRes;
    int algoCount = 0;
    // Get the algorithm
    PADDLE_ENFORCE(platform::dynload::miopenFindConvolutionBackwardDataAlgorithm(
        handle, cudnn_input_desc, input_data,cudnn_filter_desc, filter_data, cudnn_conv_desc,
        // dxDesc: Handle to the previously initialized output tensor
        // descriptor.
        cudnn_output_desc, output_data,1,&algoCount, &perfRes, cudnn_workspace,workspace_size_in_bytes,false));
    algo=perfRes.bwd_data_algo;
    // get workspace size able to allocate
    PADDLE_ENFORCE(
        platform::dynload::miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
            cudnn_output_desc, &workspace_size_in_bytes));

    // Allocate on GPU memory
    platform::CUDAPlace gpu = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    cudnn_workspace = paddle::memory::Alloc(gpu, workspace_size_in_bytes);

    // ------------------- cudnn conv transpose forward ---------------------
    int input_offset = input->numel() / input->dims()[0] / groups;
    int output_offset = output->numel() / output->dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    T alpha = 1.0f, beta = 0.0f;
    for (int g = 0; g < groups; g++) {
      PADDLE_ENFORCE(platform::dynload::miopenConvolutionBackwardData(
          handle, &alpha, cudnn_input_desc, input_data + input_offset * g,
          cudnn_filter_desc, filter_data + filter_offset * g,
          cudnn_conv_desc, algo, &beta, cudnn_output_desc, output_data + output_offset * g, cudnn_workspace,
          workspace_size_in_bytes));
    }

    // Release the cudnn workspace
    paddle::memory::Free(gpu, cudnn_workspace);
  }
};

template <typename T>
class CUDNNConvTransposeGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
#if 1
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
    miopenTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        layout, framework::vectorize2int(input->dims()));
    // Output: (N, C, O_h, O_w) or (N, C, O_d, O_h, O_w)
    miopenTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        layout, framework::vectorize2int(output_grad->dims()));
    // Filter (M, C, K_h, K_w) or (M, C, K_d K_h, K_w)
    miopenTensorDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize2int(filter->dims()));

    miopenConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

    // ------------------- cudnn backward algorithm ---------------------
    miopenConvFwdAlgorithm_t data_algo;
    miopenConvBwdWeightsAlgorithm_t filter_algo;
    size_t bwd_filter_ws_size, fwd_ws_size;
    size_t workspace_size_in_bytes = 0;
    //size_t workspace_size_limit = kConvCUDNNWorkspaceLimitBytes;
    //if (user_workspace_size > 0) {
    //  workspace_size_limit = user_workspace_size * 1024 * 1024;
    //}

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.miopen_handle();
    miopenConvAlgoPerf_t perfRes;
    void* cudnn_workspace = nullptr;
    int algoCount = 0;
    if (input_grad) {
      // choose backward algorithm for data
      PADDLE_ENFORCE(platform::dynload::miopenFindConvolutionForwardAlgorithm(
          handle, cudnn_input_desc, (const void*)input_data, cudnn_filter_desc, 
          (const void*)filter_data,cudnn_conv_desc, cudnn_output_desc, (void*)output_grad_data,
          1, &algoCount, &perfRes, (void*)cudnn_workspace, workspace_size_in_bytes, false));
      data_algo=perfRes.fwd_algo;
      PADDLE_ENFORCE(platform::dynload::miopenConvolutionForwardGetWorkSpaceSize(
          handle, cudnn_output_desc, cudnn_filter_desc, cudnn_conv_desc,
          cudnn_input_desc, &fwd_ws_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, fwd_ws_size);
    }

    if (filter_grad) {
      // choose backward algorithm for filter
      PADDLE_ENFORCE(
          platform::dynload::miopenFindConvolutionBackwardWeightsAlgorithm(
              handle, cudnn_input_desc, (const void*)input_data,cudnn_filter_desc, (const void*)filter_data,
              cudnn_conv_desc, cudnn_output_desc, (void*)output_grad_data, 1, &algoCount,
	      &perfRes, (void*)cudnn_workspace,workspace_size_in_bytes,false));
      filter_algo=perfRes.bwd_weights_algo;
      // get workspace for backwards filter algorithm
      PADDLE_ENFORCE(
          platform::dynload::miopenConvolutionBackwardWeightsGetWorkSpaceSize(
              handle, cudnn_input_desc, cudnn_output_desc, cudnn_conv_desc,
              cudnn_filter_desc, &bwd_filter_ws_size));
      workspace_size_in_bytes =
          std::max(workspace_size_in_bytes, bwd_filter_ws_size);
    }

    // ------------------- cudnn conv workspace ---------------------
    // Already on GPU
    platform::CUDAPlace gpu = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    cudnn_workspace = paddle::memory::Alloc(gpu, workspace_size_in_bytes);
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
        PADDLE_ENFORCE(platform::dynload::miopenConvolutionForward(
            handle, &alpha, cudnn_output_desc, output_grad_data + output_grad_offset * g,
            cudnn_filter_desc, filter_data + filter_offset * g, cudnn_conv_desc, data_algo,
            &beta, cudnn_input_desc, input_grad_data + input_offset * g, cudnn_workspace, 
            workspace_size_in_bytes));
      }
    }

    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      T* filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset filter_grad.
      // Gradient with respect to the filter
      for (int g = 0; g < groups; g++) {
        PADDLE_ENFORCE(platform::dynload::miopenConvolutionBackwardWeights(
            handle, &alpha, cudnn_input_desc, input_data + input_offset * g,
            cudnn_output_desc, output_grad_data + output_grad_offset * g,
            cudnn_conv_desc, filter_algo, &beta, cudnn_filter_desc,
            filter_grad_data + filter_offset * g,
            cudnn_workspace, workspace_size_in_bytes));
      }
    }

    // Release the cudnn workspace
    paddle::memory::Free(gpu, cudnn_workspace);
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(conv2d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<float>
                   /*,ops::CUDNNConvTransposeOpKernel<double>*/);
REGISTER_OP_KERNEL(conv2d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<float>
                   /*,ops::CUDNNConvTransposeGradOpKernel<double>*/);

REGISTER_OP_KERNEL(conv3d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<float>
                   /*,ops::CUDNNConvTransposeOpKernel<double>*/);
REGISTER_OP_KERNEL(conv3d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<float>
                   /*,ops::CUDNNConvTransposeGradOpKernel<double>*/);
