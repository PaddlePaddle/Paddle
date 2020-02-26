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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/padding.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;

static constexpr size_t kConvCUDNNWorkspaceLimitBytes = 1024 * 1024 * 1024;

template <typename T, int D>
static void DataTranspose(const framework::ExecutionContext& ctx,
                          const Tensor* input, Tensor* output,
                          const std::vector<int>& axis, int flag = 0) {
  auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  math::Transpose<platform::CUDADeviceContext, T, D> transpose;
  auto in_dims = input->dims();
  std::vector<int64_t> input_transpose_vec;
  for (size_t i = 0; i < axis.size(); ++i) {
    if (flag == 0)
      input_transpose_vec.push_back(in_dims[axis[i]]);
    else
      input_transpose_vec.push_back(in_dims[i]);
  }
  framework::DDim input_transpose_dims(
      framework::make_ddim(input_transpose_vec));
  output->mutable_data<T>(input_transpose_dims, ctx.GetPlace());
  transpose(dev_ctx, *input, output, axis);
}

template <typename T>
class CUDNNConvTransposeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      "It must use CUDAPlace.");
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    // cudnn v5 does not support dilations
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    int user_workspace_size = ctx.Attr<int>("workspace_size_MB");
    const T* filter_data = filter->data<T>();
    const std::string data_layout_str = ctx.Attr<std::string>("data_format");
    const paddle::operators::DataLayout data_layout =
        (data_layout_str != "NHWC" ? DataLayout::kNCHW : DataLayout::kNHWC);

    // if channel_last, transpose to channel_first
    Tensor input_transpose;
    std::vector<int> input_vec = framework::vectorize<int>(input->dims());
    std::vector<int> output_vec = framework::vectorize<int>(output->dims());
    if (data_layout == DataLayout::kNHWC) {
      if (strides.size() == 2U) {
        std::vector<int> axis = {0, 3, 1, 2};
        for (size_t i = 0; i < axis.size(); ++i) {
          input_vec[i] = input->dims()[axis[i]];
          output_vec[i] = output->dims()[axis[i]];
        }
        DataTranspose<T, 4>(ctx, input, &input_transpose, axis);
      } else if (strides.size() == 3U) {
        std::vector<int> axis = {0, 4, 1, 2, 3};
        for (size_t i = 0; i < axis.size(); ++i) {
          input_vec[i] = input->dims()[axis[i]];
          output_vec[i] = output->dims()[axis[i]];
        }
        DataTranspose<T, 5>(ctx, input, &input_transpose, axis);
      }
    } else {
      input_transpose = *input;
    }

    // update padding and dilation
    auto in_dims = input_transpose.dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    int data_dim = strides.size();  // 2d or 3d
    bool is_sys_pad = math::IsSymmetricPadding(paddings, data_dim);

    std::vector<int> input_pad(input_transpose.dims().size() * 2, 0);
    Tensor transformed_input;
    std::vector<int> padding_common(data_dim, 0);
    if (!is_sys_pad) {
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_input_shape_vec(data_dim + 2);
      new_input_shape_vec[0] = input_transpose.dims()[0];
      new_input_shape_vec[1] = input_transpose.dims()[1];

      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        new_input_shape_vec[i + 2] =
            input_transpose.dims()[i + 2] + padding_diff[i];
        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
      framework::DDim new_input_shape(
          framework::make_ddim(new_input_shape_vec));
      transformed_input.Resize(new_input_shape);
      auto& dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();

      transformed_input =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_input_shape, dev_ctx);
      const int rank = input_transpose.dims().size();
      T pad_value(0.0);
      switch (rank) {
        case 4: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, input_transpose, pad_value, &transformed_input);
        } break;
        case 5: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, input_transpose, pad_value, &transformed_input);
        } break;
        default:
          PADDLE_ENFORCE_EQ(
              rank == 4 || rank == 5, true,
              "Op(ConvTranspose) only supports 4-D or 5-D input Tensor.");
      }
    } else {
      transformed_input = input_transpose;
      if (paddings.size() == data_dim) {
        for (size_t i = 0; i < data_dim; ++i) {
          padding_common[i] = paddings[i];
        }
      } else {
        for (size_t i = 0; i < data_dim; ++i) {
          padding_common[i] = paddings[2 * i];
        }
      }
    }

    std::vector<int64_t> starts(data_dim, 0);
    std::vector<int64_t> ends(data_dim, 0);
    std::vector<int64_t> axes(data_dim, 0);
    for (size_t i = 0; i < data_dim; ++i) {
      starts[i] = input_pad[2 * i + 4] * (strides[i] + 1);
      ends[i] = starts[i] + output_vec[i + 2];
      axes[i] = i + 2;
    }

    const T* input_data = transformed_input.data<T>();
    input_vec = framework::vectorize<int>(transformed_input.dims());

    std::vector<int> transformed_output_vec = output_vec;
    for (size_t i = 0; i < data_dim; ++i) {
      transformed_output_vec[i + 2] =
          output_vec[i + 2] +
          (input_pad[2 * i + 4] + input_pad[2 * i + 5]) * strides[i] -
          2 * padding_common[i] + paddings[2 * i] + paddings[2 * i + 1];
    }

    Tensor transformed_output;
    if (!is_sys_pad) {
      DDim transformed_output_shape(
          framework::make_ddim(transformed_output_vec));
      transformed_output.mutable_data<T>(transformed_output_shape,
                                         ctx.GetPlace());
    } else {
      output->mutable_data<T>(ctx.GetPlace());
      transformed_output.ShareDataWith(*output);
      transformed_output.Resize(framework::make_ddim(transformed_output_vec));
    }
    T* transformed_output_data = transformed_output.data<T>();

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
    cudnnTensorDescriptor_t cudnn_input_desc =
        input_desc.descriptor<T>(layout, input_vec, groups);
    // (N, C, O_h, O_w) or (N, C, O_d, O_h, O_w)
    cudnnTensorDescriptor_t cudnn_output_desc =
        output_desc.descriptor<T>(layout, transformed_output_vec, groups);
    // (M, C, K_h, K_w) or (M, C, K_d, K_h, K_w)
    cudnnFilterDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize<int>(filter->dims()), groups);
    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(padding_common, strides, dilations);

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
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
            handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
            // dxDesc: Handle to the previously initialized output tensor
            // descriptor.
            cudnn_output_desc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit, &algo));

    if (FLAGS_cudnn_deterministic) {
      algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(1);
    }

    // get workspace size able to allocate
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
            cudnn_output_desc, algo, &workspace_size_in_bytes));

    // ------------------- cudnn conv transpose forward ---------------------
    int input_offset =
        transformed_input.numel() / transformed_input.dims()[0] / groups;
    int output_offset =
        transformed_output.numel() / transformed_output.dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    T alpha = static_cast<T>(1.0), beta = static_cast<T>(0.0);
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    for (int g = 0; g < groups; g++) {
      auto cudnn_func = [&](void* cudnn_workspace) {
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnConvolutionBackwardData(
                handle, &alpha, cudnn_filter_desc,
                filter_data + filter_offset * g, cudnn_input_desc,
                input_data + input_offset * g, cudnn_conv_desc, algo,
                cudnn_workspace, workspace_size_in_bytes, &beta,
                cudnn_output_desc,
                transformed_output_data + output_offset * g));
      };
      workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
    }

    if (!is_sys_pad && strides.size() == 2U) {
      Slice<paddle::platform::CUDADeviceContext, T, 4>(
          ctx, &transformed_output, output, starts, ends, axes);
    } else if (!is_sys_pad && strides.size() == 3U) {
      Slice<paddle::platform::CUDADeviceContext, T, 5>(
          ctx, &transformed_output, output, starts, ends, axes);
    }

    if (data_layout == DataLayout::kNHWC) {
      Tensor output_transpose;
      Tensor output_nchw;
      output_nchw.ShareDataWith(*output);
      output_nchw.Resize(framework::make_ddim(output_vec));
      if (strides.size() == 2U) {
        std::vector<int> axis = {0, 2, 3, 1};
        DataTranspose<T, 4>(ctx, &output_nchw, &output_transpose, axis);
        *output = output_transpose;
      } else if (strides.size() == 3U) {
        std::vector<int> axis = {0, 2, 3, 4, 1};
        DataTranspose<T, 5>(ctx, &output_nchw, &output_transpose, axis);
        *output = output_transpose;
      }
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
    const T* filter_data = filter->data<T>();

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    // cudnn v5 does not support dilations
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    int user_workspace_size = ctx.Attr<int>("workspace_size_MB");
    const std::string data_layout_str = ctx.Attr<std::string>("data_format");
    const paddle::operators::DataLayout data_layout =
        (data_layout_str != "NHWC" ? DataLayout::kNCHW : DataLayout::kNHWC);

    // if channel_last, transpose to channel_first
    Tensor input_transpose;
    Tensor output_grad_transpose;
    std::vector<int> input_vec = framework::vectorize<int>(input->dims());
    std::vector<int> output_vec =
        framework::vectorize<int>(output_grad->dims());
    if (data_layout == DataLayout::kNHWC) {
      if (strides.size() == 2U) {
        std::vector<int> axis = {0, 3, 1, 2};
        for (size_t i = 0; i < axis.size(); ++i) {
          input_vec[i] = input->dims()[axis[i]];
          output_vec[i] = output_grad->dims()[axis[i]];
        }
        DataTranspose<T, 4>(ctx, input, &input_transpose, axis);
        DataTranspose<T, 4>(ctx, output_grad, &output_grad_transpose, axis);
      } else if (strides.size() == 3U) {
        std::vector<int> axis = {0, 4, 1, 2, 3};
        for (size_t i = 0; i < axis.size(); ++i) {
          input_vec[i] = input->dims()[axis[i]];
          output_vec[i] = output_grad->dims()[axis[i]];
        }
        DataTranspose<T, 5>(ctx, input, &input_transpose, axis);
        DataTranspose<T, 5>(ctx, output_grad, &output_grad_transpose, axis);
      }
    } else {
      input_transpose = *input;
      output_grad_transpose = *output_grad;
    }

    // update padding and dilation
    auto in_dims = input_transpose.dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    int data_dim = strides.size();  // 2d or 3d
    bool is_sys_pad = math::IsSymmetricPadding(paddings, data_dim);

    std::vector<int> input_pad(input_transpose.dims().size() * 2, 0);
    Tensor transformed_output_grad;
    std::vector<int> padding_common(data_dim, 0);
    if (!is_sys_pad) {
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_output_grad_shape_vec(data_dim + 2);
      new_output_grad_shape_vec[0] = output_grad_transpose.dims()[0];
      new_output_grad_shape_vec[1] = output_grad_transpose.dims()[1];

      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        new_output_grad_shape_vec[i + 2] =
            output_grad_transpose.dims()[i + 2] + padding_diff[i];
        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
      framework::DDim new_output_grad_shape(
          framework::make_ddim(new_output_grad_shape_vec));
      transformed_output_grad.Resize(new_output_grad_shape);
      auto& dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();

      transformed_output_grad =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_output_grad_shape, dev_ctx);
      const int rank = input_transpose.dims().size();
      T pad_value(0.0);
      switch (rank) {
        case 4: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, output_grad_transpose, pad_value,
              &transformed_output_grad);
        } break;
        case 5: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, output_grad_transpose, pad_value,
              &transformed_output_grad);
        } break;
        default:
          PADDLE_ENFORCE_EQ(
              rank == 4 || rank == 5, true,
              "Op(ConvTranspose) only supports 4-D or 5-D input Tensor.");
      }
    } else {
      transformed_output_grad = output_grad_transpose;
      if (paddings.size() == data_dim) {
        for (size_t i = 0; i < data_dim; ++i) {
          padding_common[i] = paddings[i];
        }
      } else {
        for (size_t i = 0; i < data_dim; ++i) {
          padding_common[i] = paddings[2 * i];
        }
      }
    }

    const T* input_data = input_transpose.data<T>();
    const T* output_grad_data = transformed_output_grad.data<T>();
    output_vec = framework::vectorize<int>(transformed_output_grad.dims());

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

    // Input: (N, M, H, W) or (N, M, D, H, W)
    cudnnTensorDescriptor_t cudnn_input_desc =
        input_desc.descriptor<T>(layout, input_vec, groups);
    // Output: (N, C, O_h, O_w) or (N, C, O_d, O_h, O_w)
    cudnnTensorDescriptor_t cudnn_output_desc =
        output_desc.descriptor<T>(layout, output_vec, groups);
    // Filter (M, C, K_h, K_w) or (M, C, K_d K_h, K_w)
    cudnnFilterDescriptor_t cudnn_filter_desc = filter_desc.descriptor<T>(
        layout, framework::vectorize<int>(filter->dims()), groups);

    cudnnConvolutionDescriptor_t cudnn_conv_desc =
        conv_desc.descriptor<T>(padding_common, strides, dilations);

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
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardAlgorithm(
              handle, cudnn_output_desc, cudnn_filter_desc, cudnn_conv_desc,
              cudnn_input_desc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &data_algo));

      if (FLAGS_cudnn_deterministic) {
        data_algo = static_cast<cudnnConvolutionFwdAlgo_t>(1);
      }
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionForwardWorkspaceSize(
              handle, cudnn_output_desc, cudnn_filter_desc, cudnn_conv_desc,
              cudnn_input_desc, data_algo, &fwd_ws_size));
      workspace_size_in_bytes = std::max(workspace_size_in_bytes, fwd_ws_size);
    }

    if (filter_grad) {
      // choose backward algorithm for filter
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              handle, cudnn_output_desc, cudnn_input_desc, cudnn_conv_desc,
              cudnn_filter_desc,
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit, &filter_algo));

      if (FLAGS_cudnn_deterministic) {
        filter_algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(1);
      }
      // get workspace for backwards filter algorithm
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
              handle, cudnn_output_desc, cudnn_input_desc, cudnn_conv_desc,
              cudnn_filter_desc, filter_algo, &bwd_filter_ws_size));
      workspace_size_in_bytes =
          std::max(workspace_size_in_bytes, bwd_filter_ws_size);
    }

    // ------------------- cudnn conv backward data ---------------------
    // FIXME(typhoonzero): template type T may not be the same as cudnn call.
    int input_offset = input->numel() / input->dims()[0] / groups;
    int output_grad_offset = transformed_output_grad.numel() /
                             transformed_output_grad.dims()[0] / groups;
    int filter_offset = filter->numel() / groups;
    T alpha = static_cast<T>(1.0), beta = static_cast<T>(0.0);
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    if (input_grad) {
      T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset input_grad.
      for (int g = 0; g < groups; g++) {
        auto cudnn_func = [&](void* cudnn_workspace) {
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnConvolutionForward(
                  handle, &alpha, cudnn_output_desc,
                  output_grad_data + output_grad_offset * g, cudnn_filter_desc,
                  filter_data + filter_offset * g, cudnn_conv_desc, data_algo,
                  cudnn_workspace, workspace_size_in_bytes, &beta,
                  cudnn_input_desc, input_grad_data + input_offset * g));
        };
        workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
      }

      if (data_layout == DataLayout::kNHWC) {
        Tensor input_grad_transpose;
        Tensor input_grad_nchw;
        input_grad_nchw.ShareDataWith(*input_grad);
        input_grad_nchw.Resize(framework::make_ddim(input_vec));
        if (strides.size() == 2U) {
          std::vector<int> axis = {0, 2, 3, 1};
          DataTranspose<T, 4>(ctx, &input_grad_nchw, &input_grad_transpose,
                              axis);
          *input_grad = input_grad_transpose;
        } else if (strides.size() == 3U) {
          std::vector<int> axis = {0, 2, 3, 4, 1};
          DataTranspose<T, 5>(ctx, &input_grad_nchw, &input_grad_transpose,
                              axis);
          *input_grad = input_grad_transpose;
        }
      }
    }

    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      T* filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace());
      // Because beta is zero, it is unnecessary to reset filter_grad.
      // Gradient with respect to the filter
      for (int g = 0; g < groups; g++) {
        auto cudnn_func = [&](void* cudnn_workspace) {
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnConvolutionBackwardFilter(
                  handle, &alpha, cudnn_output_desc,
                  output_grad_data + output_grad_offset * g, cudnn_input_desc,
                  input_data + input_offset * g, cudnn_conv_desc, filter_algo,
                  cudnn_workspace, workspace_size_in_bytes, &beta,
                  cudnn_filter_desc, filter_grad_data + filter_offset * g));
        };
        workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_KERNEL(conv2d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeOpKernel<float>,
                   ops::CUDNNConvTransposeOpKernel<double>);
REGISTER_OP_KERNEL(conv2d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeGradOpKernel<float>,
                   ops::CUDNNConvTransposeGradOpKernel<double>);

REGISTER_OP_KERNEL(conv3d_transpose, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeOpKernel<float>,
                   ops::CUDNNConvTransposeOpKernel<double>);
REGISTER_OP_KERNEL(conv3d_transpose_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNConvTransposeGradOpKernel<plat::float16>,
                   ops::CUDNNConvTransposeGradOpKernel<float>,
                   ops::CUDNNConvTransposeGradOpKernel<double>);
