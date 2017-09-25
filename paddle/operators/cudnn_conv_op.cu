/* Copyright (c) 2016 PaddlePaddle Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/op_registry.h"
#include "paddle/memory/memory.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;
using CUDADeviceContext = platform::CUDADeviceContext;

// NOTE: framework::vectorize converts to type int64_t
//       which does not fit cudnn inputs.
std::vector<int> dims2vector(const framework::DDim& dims) {
  std::vector<int> ret;
  for (int i = 0; i < dims.size(); i++) {
    ret.push_back(dims[i]);
  }
  return ret;
}

template <typename T>
class CudnnConvOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedFilterDescriptor filter_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout = DataLayout::kNCHW;

    cudnnTensorDescriptor_t cudnn_input_desc =
        input_desc.descriptor<T>(layout, dims2vector(input->dims()), groups);
    cudnnTensorDescriptor_t cudnn_output_desc =
        output_desc.descriptor<T>(layout, dims2vector(output->dims()), groups);
    cudnnFilterDescriptor_t cudnn_filter_desc =
        filter_desc.descriptor<T>(layout, dims2vector(filter->dims()), groups);
    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

    int input_channels = input->dims()[1];
    int input_height = input->dims()[2];
    int input_width = input->dims()[3];
    int output_channels = output->dims()[1];
    int output_height = output->dims()[2];
    int output_width = output->dims()[3];

    int group_offset_X = input_channels / groups * input_height * input_width;
    int group_offset_Y =
        output_channels / groups * output_height * output_width;
    int group_offset_filter = filter->numel() / groups;
    // ------------------- cudnn conv algorithm ---------------------
    cudnnConvolutionFwdAlgo_t algo;
    auto h = ctx.cuda_device_context().cudnn_handle();

    PADDLE_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm(
        h, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
        cudnn_output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
    // ------------------- cudnn conv workspace ---------------------
    void* cudnn_workspace = nullptr;
    size_t workspace_size_in_bytes = 0;
    PADDLE_ENFORCE(platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
        h, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
        cudnn_output_desc, algo, &workspace_size_in_bytes));
    // Already on GPU
    platform::GPUPlace gpu = boost::get<platform::GPUPlace>(ctx.GetPlace());
    cudnn_workspace = paddle::memory::Alloc(gpu, workspace_size_in_bytes);
    // ------------------- cudnn conv forward ---------------------
    // FIXME(typhoonzero): template type T may not be the same as cudnn call.
    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < groups; i++) {
      PADDLE_ENFORCE(platform::dynload::cudnnConvolutionForward(
          h, &alpha, cudnn_input_desc, input_data + i * group_offset_X,
          cudnn_filter_desc, filter_data + i * group_offset_filter,
          cudnn_conv_desc, algo, cudnn_workspace, workspace_size_in_bytes,
          &beta, cudnn_output_desc, output_data + i * group_offset_Y));
    }
    // Release the cudnn workspace
    paddle::memory::Free(gpu, cudnn_workspace);
  }
};

template <typename T>
class CudnnConvGradOpKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
    // input is actually the output of forward.
    auto input = ctx.Input<Tensor>("Input");
    // filter is the filter used in forward.
    auto filter = ctx.Input<Tensor>("Filter");
    // output_grad is gradient output of the operator connected after it.
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    // input_grad is the gradient output of conv op.
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    // filter_grad is the gradient output of filter(kernel).
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    const T* input_data = input->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    const T* filter_data = filter->data<T>();

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_grad_desc;
    ScopedTensorDescriptor input_grad_desc;

    ScopedFilterDescriptor filter_desc;
    ScopedFilterDescriptor filter_grad_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout = DataLayout::kNCHW;

    cudnnTensorDescriptor_t cudnn_input_desc =
        input_desc.descriptor<T>(layout, dims2vector(input->dims()), groups);
    cudnnTensorDescriptor_t cudnn_output_grad_desc =
        output_grad_desc.descriptor<T>(layout, dims2vector(output_grad->dims()),
                                       groups);
    cudnnFilterDescriptor_t cudnn_filter_desc =
        filter_desc.descriptor<T>(layout, dims2vector(filter->dims()), groups);
    cudnnTensorDescriptor_t cudnn_input_grad_desc = nullptr;
    cudnnFilterDescriptor_t cudnn_filter_grad_desc = nullptr;

    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

    // int n_dims = input->dims().size();
    int input_channels = input->dims()[1];
    int input_height = input->dims()[2];
    int input_width = input->dims()[3];
    // int output_grad_dims = output_grad->dims().size();
    int output_grad_channels = filter->dims()[0];
    int output_grad_height = output_grad->dims()[2];
    int output_grad_width = output_grad->dims()[3];

    int group_offset_X = input_channels / groups * input_height * input_width;
    int group_offset_Y =
        output_grad_channels / groups * output_grad_height * output_grad_width;
    int group_offset_filter = filter->numel() / groups;
    // ------------------- cudnn backward algorithm ---------------------
    cudnnConvolutionBwdDataAlgo_t data_algo;
    cudnnConvolutionBwdFilterAlgo_t filter_algo;
    size_t workspace_size_in_bytes = 0, tmp_size = 0;
    auto h = ctx.cuda_device_context().cudnn_handle();
    if (input_grad) {
      cudnn_input_grad_desc = input_grad_desc.descriptor<T>(
          layout, dims2vector(input_grad->dims()), groups);
      PADDLE_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
              h, cudnn_filter_desc,
              // dyDesc: Handle to the previously initialized input differential
              // tensor descriptor.
              cudnn_output_grad_desc, cudnn_conv_desc,
              // dxDesc: Handle to the previously initialized output tensor
              // descriptor.
              cudnn_input_grad_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
              0, &data_algo));
      PADDLE_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
              h, cudnn_filter_desc, cudnn_output_grad_desc, cudnn_conv_desc,
              cudnn_input_grad_desc, data_algo, &tmp_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, tmp_size);
    }

    if (filter_grad) {
      cudnn_filter_grad_desc = filter_grad_desc.descriptor<T>(
          layout, dims2vector(filter_grad->dims()), groups);
      PADDLE_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              h, cudnn_input_desc, cudnn_output_grad_desc, cudnn_conv_desc,
              cudnn_filter_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
              &filter_algo));

      PADDLE_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
              h, cudnn_input_desc, cudnn_output_grad_desc, cudnn_conv_desc,
              cudnn_filter_desc, filter_algo, &tmp_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, tmp_size);
    }
    // ------------------- cudnn conv workspace ---------------------
    // Already on GPU
    void* cudnn_workspace = nullptr;
    platform::GPUPlace gpu = boost::get<platform::GPUPlace>(ctx.GetPlace());
    cudnn_workspace = paddle::memory::Alloc(gpu, workspace_size_in_bytes);
    // ------------------- cudnn conv backward data ---------------------
    // FIXME(typhoonzero): template type T may not be the same as cudnn call.
    float alpha = 1.0f, beta = 0.0f;
    if (input_grad) {
      for (int i = 0; i < groups; i++) {
        T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
        PADDLE_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
            h, &alpha, cudnn_filter_desc, filter_data + i * group_offset_filter,
            cudnn_output_grad_desc, output_grad_data + i * group_offset_Y,
            cudnn_conv_desc, data_algo, cudnn_workspace,
            workspace_size_in_bytes, &beta, cudnn_input_grad_desc,
            input_grad_data + i * group_offset_X));
      }
    }
    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      for (int i = 0; i < groups; i++) {
        T* filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
        PADDLE_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
            h, &alpha, cudnn_input_desc, input_data + i * group_offset_X,
            cudnn_output_grad_desc, output_grad_data + i * group_offset_Y,
            cudnn_conv_desc, filter_algo, cudnn_workspace,
            workspace_size_in_bytes, &beta, cudnn_filter_grad_desc,
            filter_grad_data + i * group_offset_filter));
      }
    }
    // Release the cudnn workspace
    paddle::memory::Free(gpu, cudnn_workspace);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_GPU_KERNEL(cudnn_conv, paddle::operators::CudnnConvOpKernel<float>);
REGISTER_OP_GPU_KERNEL(cudnn_conv_grad,
                       paddle::operators::CudnnConvGradOpKernel<float>);
