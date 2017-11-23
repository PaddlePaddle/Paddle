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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/memory/memory.h"
#include "paddle/operators/conv_op.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;

static constexpr size_t kCONV_CUDNN_WORKSPACE_LIMIT_BYTES = 1024 * 1024 * 1024;

inline void SetTensorGroups(std::vector<int>& dims, int groups) {
  // std::vector<int> dims_with_group(dims.begin(), dims.end());  // copy
  if (groups > 1) {
    dims[1] /= groups;
  }
}

inline void SetFilterGroups(std::vector<int>& dims, int groups) {
  // std::vector<int> kernel_with_group(kernel.begin(), kernel.end());
  if (groups > 1) {
    // M /= groups
    dims[0] /= groups;
  }
}

template <typename T>
class CudnnConvOpKernel : public framework::OpKernel<T> {
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
    int user_workspace_size = ctx.Attr<int>("workspace_size_MB");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    ScopedFilterDescriptor filter_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout = DataLayout::kNCHW;
    if (input->dims().size() == 5) {
      layout = DataLayout::kNCDHW;
    }

    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

#if CUDNN_VERSION > 6000
    // cudnn 7 can support groups, no need to do it mannually
    // FIXME(typhoonzero): find a better way to disable groups
    // rather than setting it to 1.
    PADDLE_ENFORCE(cudnnSetConvolutionGroupCount(cudnn_conv_desc, groups));
    groups = 1;
#endif

    auto input_dim_vec = framework::vectorize2int(input->dims());
    SetTensorGroups(input_dim_vec, groups);
    cudnnTensorDescriptor_t cudnn_input_desc =
        input_desc.descriptor<T>(layout, input_dim_vec);
    auto output_dim_vec = framework::vectorize2int(output->dims());
    SetTensorGroups(output_dim_vec, groups);
    cudnnTensorDescriptor_t cudnn_output_desc =
        output_desc.descriptor<T>(layout, output_dim_vec);
    auto filter_dim_vec = framework::vectorize2int(filter->dims());
    SetFilterGroups(filter_dim_vec, groups);
    cudnnFilterDescriptor_t cudnn_filter_desc =
        filter_desc.descriptor<T>(layout, filter_dim_vec);

    int input_channels = input->dims()[1];
    int input_height, input_width, input_depth;
    if (input->dims().size() == 5) {
      input_depth = input->dims()[2];
      input_height = input->dims()[3];
      input_width = input->dims()[4];
    } else {
      input_depth = 1;
      input_height = input->dims()[2];
      input_width = input->dims()[3];
    }
    int output_channels = output->dims()[1];
    int output_height, output_width, output_depth;
    if (output->dims().size() == 5) {
      output_depth = output->dims()[2];
      output_height = output->dims()[3];
      output_width = output->dims()[4];
    } else {  // dim size is enforced in InferShape
      output_depth = 1;
      output_height = output->dims()[2];
      output_width = output->dims()[3];
    }

    int group_offset_in =
        input_channels / groups * input_height * input_width * input_depth;
    int group_offset_out =
        output_channels / groups * output_height * output_width * output_depth;
    int group_offset_filter = filter->numel() / groups;
    // ------------------- cudnn conv workspace ---------------------
    void* cudnn_workspace = nullptr;
    size_t workspace_size_in_bytes;  // final workspace to allocate.
    size_t workspace_size_limit = kCONV_CUDNN_WORKSPACE_LIMIT_BYTES;
    if (user_workspace_size > 0) {
      workspace_size_limit = user_workspace_size * 1024 * 1024;
    }
    // ------------------- cudnn conv algorithm ---------------------
    cudnnConvolutionFwdAlgo_t algo;
    auto handle = ctx.cuda_device_context().cudnn_handle();

    PADDLE_ENFORCE(platform::dynload::cudnnGetConvolutionForwardAlgorithm(
        handle, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
        cudnn_output_desc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_size_limit, &algo));
    // get workspace size able to allocate
    PADDLE_ENFORCE(platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
        handle, cudnn_input_desc, cudnn_filter_desc, cudnn_conv_desc,
        cudnn_output_desc, algo, &workspace_size_in_bytes));
    // Allocate on GPU memory
    platform::GPUPlace gpu = boost::get<platform::GPUPlace>(ctx.GetPlace());
    cudnn_workspace = paddle::memory::Alloc(gpu, workspace_size_in_bytes);
    // ------------------- cudnn conv forward ---------------------
    T alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < groups; i++) {
      PADDLE_ENFORCE(platform::dynload::cudnnConvolutionForward(
          handle, &alpha, cudnn_input_desc, input_data + i * group_offset_in,
          cudnn_filter_desc, filter_data + i * group_offset_filter,
          cudnn_conv_desc, algo, cudnn_workspace, workspace_size_in_bytes,
          &beta, cudnn_output_desc, output_data + i * group_offset_out));
    }
    // Release the cudnn workspace
    paddle::memory::Free(gpu, cudnn_workspace);
  }
};

template <typename T>
class CudnnConvGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
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
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    int user_workspace_size = ctx.Attr<int>("workspace_size_MB");

    // ------------------- cudnn descriptors ---------------------
    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_grad_desc;
    ScopedTensorDescriptor input_grad_desc;

    ScopedFilterDescriptor filter_desc;
    ScopedFilterDescriptor filter_grad_desc;
    ScopedConvolutionDescriptor conv_desc;
    DataLayout layout = DataLayout::kNCHW;
    if (input->dims().size() == 5) {
      layout = DataLayout::kNCDHW;
    }

    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(paddings, strides, dilations);

#if CUDNN_VERSION > 6000
    // cudnn 7 can support groups, no need to do it mannually
    // FIXME(typhoonzero): find a better way to disable groups
    // rather than setting it to 1.
    PADDLE_ENFORCE(cudnnSetConvolutionGroupCount(cudnn_conv_desc, groups));
    groups = 1;
#endif

    auto input_dim_vec = framework::vectorize2int(input->dims());
    SetTensorGroups(input_dim_vec, groups);
    cudnnTensorDescriptor_t cudnn_input_desc =
        input_desc.descriptor<T>(layout, input_dim_vec);
    auto output_grad_dim_vec = framework::vectorize2int(output_grad->dims());
    SetTensorGroups(output_grad_dim_vec, groups);
    cudnnTensorDescriptor_t cudnn_output_grad_desc =
        output_grad_desc.descriptor<T>(layout, output_grad_dim_vec);
    auto filter_dims_vec = framework::vectorize2int(filter->dims());
    SetFilterGroups(filter_dims_vec, groups);
    cudnnFilterDescriptor_t cudnn_filter_desc =
        filter_desc.descriptor<T>(layout, filter_dims_vec);
    cudnnTensorDescriptor_t cudnn_input_grad_desc = nullptr;
    cudnnFilterDescriptor_t cudnn_filter_grad_desc = nullptr;

    int input_channels = input->dims()[1];
    int input_height, input_width, input_depth;
    if (input->dims().size() == 5) {
      input_depth = input->dims()[2];
      input_height = input->dims()[3];
      input_width = input->dims()[4];
    } else {  // dim size is enforced in InferShape
      input_depth = 1;
      input_height = input->dims()[2];
      input_width = input->dims()[3];
    }

    int output_grad_channels = filter->dims()[0];
    int output_grad_height, output_grad_width, output_grad_depth;
    if (output_grad->dims().size() == 5) {
      output_grad_depth = output_grad->dims()[2];
      output_grad_height = output_grad->dims()[3];
      output_grad_width = output_grad->dims()[4];
    } else {
      output_grad_depth = 1;
      output_grad_height = output_grad->dims()[2];
      output_grad_width = output_grad->dims()[3];
    }

    int group_offset_in =
        input_channels / groups * input_height * input_width * input_depth;
    int group_offset_out = output_grad_channels / groups * output_grad_height *
                           output_grad_width * output_grad_depth;
    int group_offset_filter = filter->numel() / groups;
    // ------------------- cudnn backward algorithm ---------------------
    cudnnConvolutionBwdDataAlgo_t data_algo;
    cudnnConvolutionBwdFilterAlgo_t filter_algo;
    size_t workspace_size_in_bytes = 0, tmp_size = 0;
    size_t workspace_size_limit = kCONV_CUDNN_WORKSPACE_LIMIT_BYTES;
    if (user_workspace_size > 0) {
      workspace_size_limit = user_workspace_size * 1024 * 1024;
    }

    auto handle = ctx.cuda_device_context().cudnn_handle();
    if (input_grad) {
      auto input_grad_dim_vec = framework::vectorize2int(input_grad->dims());
      SetTensorGroups(input_grad_dim_vec, groups);
      cudnn_input_grad_desc =
          input_grad_desc.descriptor<T>(layout, input_grad_dim_vec);
      PADDLE_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
              handle, cudnn_filter_desc,
              // dyDesc: Handle to the previously initialized input differential
              // tensor descriptor.
              cudnn_output_grad_desc, cudnn_conv_desc,
              // dxDesc: Handle to the previously initialized output tensor
              // descriptor.
              cudnn_input_grad_desc,
              CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &data_algo));
      PADDLE_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
              handle, cudnn_filter_desc, cudnn_output_grad_desc,
              cudnn_conv_desc, cudnn_input_grad_desc, data_algo, &tmp_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, tmp_size);
    }

    if (filter_grad) {
      auto filter_grad_dim_vec = framework::vectorize2int(filter_grad->dims());
      SetFilterGroups(filter_grad_dim_vec, groups);
      cudnn_filter_grad_desc =
          filter_grad_desc.descriptor<T>(layout, filter_grad_dim_vec);
      PADDLE_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              handle, cudnn_input_desc, cudnn_output_grad_desc, cudnn_conv_desc,
              cudnn_filter_desc,
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &filter_algo));

      PADDLE_ENFORCE(
          platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
              handle, cudnn_input_desc, cudnn_output_grad_desc, cudnn_conv_desc,
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
    T alpha = 1.0f, beta = 0.0f;
    if (input_grad) {
      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset input_grad.

      for (int i = 0; i < groups; i++) {
        PADDLE_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
            handle, &alpha, cudnn_filter_desc,
            filter_data + i * group_offset_filter, cudnn_output_grad_desc,
            output_grad_data + i * group_offset_out, cudnn_conv_desc, data_algo,
            cudnn_workspace, workspace_size_in_bytes, &beta,
            cudnn_input_grad_desc, input_grad_data + i * group_offset_in));
      }
    }
    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      T* filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset filter_grad.

      for (int i = 0; i < groups; i++) {
        PADDLE_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
            handle, &alpha, cudnn_input_desc, input_data + i * group_offset_in,
            cudnn_output_grad_desc, output_grad_data + i * group_offset_out,
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

REGISTER_OP_GPU_KERNEL(conv_cudnn, paddle::operators::CudnnConvOpKernel<float>);
REGISTER_OP_GPU_KERNEL(conv_cudnn_grad,
                       paddle::operators::CudnnConvGradOpKernel<float>);
