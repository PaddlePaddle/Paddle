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
#include "paddle/fluid/operators/conv_cudnn_helper.h"
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/operators/math/padding.h"
#include "paddle/fluid/platform/cudnn_helper.h"
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
template <typename T>
using ScalingParamType = typename platform::CudnnDataType<T>::ScalingParamType;
using framework::AlgorithmsCache;

static inline void GetNCDHW(const framework::DDim& dims,
                            const DataLayout& layout, int* N, int* C, int* D,
                            int* H, int* W) {
  *N = dims[0];
  *C = layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int i = layout == DataLayout::kNCHW ? 0 : 1;
  if (dims.size() == 5) {
    *D = dims[2 - i];
    *H = dims[3 - i];
    *W = dims[4 - i];
  } else {
    *D = 1;
    *H = dims[2 - i];
    *W = dims[3 - i];
  }
}

static inline bool IsSymmetricPadding(const std::vector<int>& paddings,
                                      const int data_dim) {
  bool is_sys_pad = true;
  if (paddings.size() == data_dim * 2) {
    for (size_t i = 0; i < data_dim; ++i) {
      if (paddings[2 * i] != paddings[2 * i + 1]) {
        is_sys_pad = false;
        return is_sys_pad;
      }
    }
  }
  return is_sys_pad;
}

template <typename DeviceContext, typename T, size_t D>
static void Slice_2(const framework::ExecutionContext& context,
                    const Tensor* input, Tensor* out,
                    const std::vector<int>& starts,
                    const std::vector<int>& axes) {
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  auto in_dims = input->dims();
  auto new_out_dims = out->dims();
  auto offsets = Eigen::array<int, D>();
  auto extents = Eigen::array<int, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = new_out_dims[i];
  }

  int start;
  for (size_t i = 0; i < axes.size(); ++i) {
    start = starts[i];
    if (start < 0) {
      start = (start + in_dims[axes[i]]);
    }
    start = std::max(start, 0);
    offsets[axes[i]] = start;
  }
  auto in_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *input);

  auto out_t =
      framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
          *out, new_out_dims);
  out_t.device(place) = in_t.slice(offsets, extents);
}

template <typename T>
class CUDNNConvOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      "It must use CUDAPlace.");
    const Tensor* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");

    if (exhaustive_search && FLAGS_cudnn_deterministic) {
      PADDLE_THROW(
          "Cann't set exhaustive_search True and "
          "FLAGS_cudnn_deterministic True at same time.");
    }
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // ------------ transformed tensor -----------
    Tensor transformed_input_channel(input->type());
    Tensor transformed_output(output->type());
    T* output_data = nullptr;
    if (channel_last) {
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, input, &transformed_input_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, input, &transformed_input_channel);

      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, output,
                                                           &transformed_output);

    } else {
      transformed_input_channel = *input;
      transformed_output = *output;
    }
    output_data = transformed_output.data<T>();

    // update padding and dilation
    auto in_dims = transformed_input_channel.dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());

    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    int data_dim = strides.size();  // 2d or 3d
    bool is_sys_pad = IsSymmetricPadding(paddings, data_dim);

    Tensor transformed_input;
    std::vector<int> padding_common(data_dim, 0);
    if (!is_sys_pad) {
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_input_shape_vec(data_dim + 2);
      new_input_shape_vec[0] = transformed_input_channel.dims()[0];
      new_input_shape_vec[1] = transformed_input_channel.dims()[1];

      std::vector<int> input_pad(transformed_input_channel.dims().size() * 2,
                                 0);
      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        new_input_shape_vec[i + 2] =
            transformed_input_channel.dims()[i + 2] + padding_diff[i];
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
      const int rank = transformed_input_channel.dims().size();
      T pad_value(0.0);
      switch (rank) {
        case 4: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, transformed_input_channel, pad_value,
              &transformed_input);
        } break;
        case 5: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, transformed_input_channel, pad_value,
              &transformed_input);
        } break;
        default:
          PADDLE_THROW("ConvOp only support tensors with 4 or 5 dimensions.");
      }

    } else {
      transformed_input = transformed_input_channel;
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

    const T* input_data = transformed_input.data<T>();
    const T* filter_data = filter->data<T>();

    // ------------------- cudnn descriptors ---------------------
    ConvArgs args{&transformed_input, filter,   &transformed_output, strides,
                  padding_common,     dilations};

    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    auto dtype = platform::CudnnDataType<T>::type;
    DataLayout layout = DataLayout::kNCHW;
    if (transformed_input_channel.dims().size() == 5) {
      layout = DataLayout::kNCDHW;
    }
    auto layout_format = GetCudnnTensorFormat(layout);

    args.handle = handle;
    args.cdesc.set(dtype, padding_common, strides, dilations);

#if CUDNN_VERSION_MIN(7, 0, 1)
    // cudnn 7 can support groups, no need to do it manually
    // FIXME(typhoonzero): find a better way to disable groups
    // rather than setting it to 1.
    CUDNN_ENFORCE(platform::dynload::cudnnSetConvolutionGroupCount(
        args.cdesc.desc(), groups));
    groups = 1;
#endif
    args.idesc.set(transformed_input, groups);

    args.wdesc.set(*filter, layout_format, groups);
    args.odesc.set(transformed_output, groups);
    int i_n, i_c, i_d, i_h, i_w;

    GetNCDHW(transformed_input.dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d,
             &i_h, &i_w);
    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(transformed_output.dims(), DataLayout::kNCHW, &o_n, &o_c, &o_d,
             &o_h, &o_w);

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
    int group_offset_filter = filter->numel() / groups;
    // ------------------- cudnn conv workspace ---------------------
    size_t workspace_size = 0;  // final workspace to allocate.
    // ------------------- cudnn conv algorithm ---------------------
    cudnnConvolutionFwdAlgo_t algo{};

    using search = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
    algo = search::Find<T>(args, exhaustive_search, false, 0, ctx);
    workspace_size = search::GetWorkspaceSize(args, algo);

    // ------------------- cudnn conv forward ---------------------
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < groups; i++) {
      workspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
                handle, &alpha, args.idesc.desc(),
                input_data + i * group_offset_in, args.wdesc.desc(),
                filter_data + i * group_offset_filter, args.cdesc.desc(), algo,
                workspace_ptr, workspace_size, &beta, args.odesc.desc(),
                output_data + i * group_offset_out));
          },
          workspace_size);
    }

    if (channel_last) {
      TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
          ctx, &transformed_output, output);
    }
  }
};

template <typename T>
class CUDNNConvGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      "It must use CUDAPlace.");
    auto input = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    const T* filter_data = filter->data<T>();
    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
    }
    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
    }

    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    int groups = ctx.Attr<int>("groups");
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");
    bool deterministic = FLAGS_cudnn_deterministic;
    if (exhaustive_search && deterministic) {
      PADDLE_THROW(
          "Can't set exhaustive_search True and "
          "FLAGS_cudnn_deterministic True at same time.");
    }
    const std::string data_format = ctx.Attr<std::string>("data_format");
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // transform Tensor
    Tensor transformed_input_channel(input->type());
    Tensor transformed_output_grad_channel(output_grad->type());
    Tensor transformed_input_grad_channel(input->type());

    if (channel_last) {
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, input, &transformed_input_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, input, &transformed_input_channel);

      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, output_grad, &transformed_output_grad_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, output_grad, &transformed_output_grad_channel);

      if (input_grad) {
        ResizeToChannelFirst<platform::CUDADeviceContext, T>(
            ctx, input_grad, &transformed_input_grad_channel);
      }

    } else {
      transformed_input_channel = *input;
      transformed_output_grad_channel = *output_grad;
      if (input_grad) {
        transformed_input_grad_channel.ShareDataWith(*input_grad);
      }
    }

    //  update paddings
    auto in_dims = transformed_input_channel.dims();
    auto filter_dims = filter->dims();
    framework::DDim in_data_dims;
    in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    // cuDNN only supports padding the same amount on every dimension.
    // So we create a new padded input tensor.
    int data_dim = strides.size();  // 2d or 3d
    bool is_sys_pad = IsSymmetricPadding(paddings, data_dim);
    Tensor transformed_input(input->type());
    Tensor transformed_input_grad(input->type());
    std::vector<int> padding_common(data_dim, 0);
    std::vector<int> input_pad(transformed_input_channel.dims().size() * 2, 0);

    if (!is_sys_pad) {
      // get pad
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_input_shape_vec(data_dim + 2);
      new_input_shape_vec[0] = transformed_input_channel.dims()[0];
      new_input_shape_vec[1] = transformed_input_channel.dims()[1];

      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        new_input_shape_vec[i + 2] =
            transformed_input_channel.dims()[i + 2] + padding_diff[i];
        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
      framework::DDim new_input_shape(
          framework::make_ddim(new_input_shape_vec));
      transformed_input.Resize(new_input_shape);

      transformed_input_grad.Resize(new_input_shape);
      auto& dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();

      transformed_input =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_input_shape, dev_ctx);
      if (input_grad) {
        transformed_input_grad =
            ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
                new_input_shape, dev_ctx);
      }
      // pad for input
      const int rank = transformed_input_channel.dims().size();
      T pad_value(0.0);
      switch (rank) {
        case 4: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, transformed_input_channel, pad_value,
              &transformed_input);
        } break;
        case 5: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, transformed_input_channel, pad_value,
              &transformed_input);
        } break;
        default:
          PADDLE_THROW("ConvOp only support tensors with 4 or 5 dimensions.");
      }
    } else {
      transformed_input.ShareDataWith(transformed_input_channel);
      if (input_grad) {
        transformed_input_grad.ShareDataWith(transformed_input_grad_channel);
      }
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

    const T* input_data = transformed_input.data<T>();
    const T* output_grad_data = transformed_output_grad_channel.data<T>();
    T* filter_grad_data = nullptr;
    T* input_grad_data = nullptr;
    T* transformed_input_grad_data = nullptr;

    ConvArgs args1{&transformed_input_grad,
                   filter,
                   &transformed_output_grad_channel,
                   strides,
                   padding_common,
                   dilations};
    ConvArgs args2{&transformed_input,
                   filter_grad,
                   &transformed_output_grad_channel,
                   strides,
                   padding_common,
                   dilations};

    auto handle = dev_ctx.cudnn_handle();
    auto dtype = platform::CudnnDataType<T>::type;
    DataLayout layout = DataLayout::kNCHW;
    if (input->dims().size() == 5) {
      layout = DataLayout::kNCDHW;
    }
    auto layout_tensor = GetCudnnTensorFormat(layout);
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();

    int i_n, i_c, i_d, i_h, i_w;
    GetNCDHW(transformed_input.dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d,
             &i_h, &i_w);
    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(transformed_output_grad_channel.dims(), DataLayout::kNCHW, &o_n,
             &o_c, &o_d, &o_h, &o_w);

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
    int group_offset_filter = filter->numel() / groups;
    // ------------------- cudnn backward algorithm ---------------------
    cudnnConvolutionBwdDataAlgo_t data_algo =
        static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
    cudnnConvolutionBwdFilterAlgo_t filter_algo =
        static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);
    size_t workspace_size = 0;
    int iwo_groups, c_groups;

#if CUDNN_VERSION_MIN(7, 0, 1)
    iwo_groups = 1;
    c_groups = groups;
    groups = 1;
#endif

    if (input_grad) {
      // ------------------- cudnn descriptors ---------------------
      input_grad_data = input_grad->data<T>();
      transformed_input_grad_data = transformed_input_grad.data<T>();
      args1.handle = handle;
      args1.idesc.set(transformed_input_grad, iwo_groups);
      args1.wdesc.set(*filter, layout_tensor, iwo_groups);
      args1.odesc.set(transformed_output_grad_channel, iwo_groups);
      args1.cdesc.set(dtype, padding_common, strides, dilations, c_groups);

      using search1 = SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
      data_algo =
          search1::Find<T>(args1, exhaustive_search, deterministic, 0, ctx);
      workspace_size =
          std::max(workspace_size, search1::GetWorkspaceSize(args1, data_algo));
    }

    if (filter_grad) {
      // ------------------- cudnn descriptors ---------------------
      filter_grad_data = filter_grad->data<T>();
      args2.handle = handle;
      args2.idesc.set(transformed_input, iwo_groups);
      args2.wdesc.set(*filter_grad, layout_tensor, iwo_groups);
      args2.odesc.set(transformed_output_grad_channel, iwo_groups);
      args2.cdesc.set(dtype, padding_common, strides, dilations, c_groups);

      using search2 = SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
      filter_algo =
          search2::Find<T>(args2, exhaustive_search, deterministic, 1, ctx);
      workspace_size = std::max(workspace_size,
                                search2::GetWorkspaceSize(args2, filter_algo));
    }

    // ------------------- cudnn conv backward data ---------------------
    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    if (input_grad) {
      // Because beta is zero, it is unnecessary to reset input_grad.
      for (int i = 0; i < groups; i++) {
        workspace_handle.RunFunc(
            [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
                  handle, &alpha, args1.wdesc.desc(),
                  filter_data + i * group_offset_filter, args1.odesc.desc(),
                  output_grad_data + i * group_offset_out, args1.cdesc.desc(),
                  data_algo, cudnn_workspace_ptr, workspace_size, &beta,
                  args1.idesc.desc(),
                  transformed_input_grad_data + i * group_offset_in));
            },
            workspace_size);
      }

      std::vector<int> starts(transformed_input_channel.dims().size(), 0);
      std::vector<int> axes(transformed_input_channel.dims().size(), 0);

      for (size_t i = 0; i < transformed_input_channel.dims().size(); ++i) {
        starts[i] = input_pad[2 * i];
        axes[i] = i;
      }

      transformed_input_grad_channel.mutable_data(ctx.GetPlace());
      if (transformed_input_channel.dims().size() == 4) {
        Slice_2<paddle::platform::CUDADeviceContext, T, 4>(
            ctx, &transformed_input_grad, &transformed_input_grad_channel,
            starts, axes);
      } else {
        Slice_2<paddle::platform::CUDADeviceContext, T, 5>(
            ctx, &transformed_input_grad, &transformed_input_grad_channel,
            starts, axes);
      }

      if (channel_last) {
        TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
            ctx, &transformed_input_grad_channel, input_grad);
      }
    }
    // ------------------- cudnn conv backward filter ---------------------
    if (filter_grad) {
      // Because beta is zero, it is unnecessary to reset filter_grad.
      for (int i = 0; i < groups; i++) {
        workspace_handle.RunFunc(
            [&](void* cudnn_workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
                  handle, &alpha, args2.idesc.desc(),
                  input_data + i * group_offset_in, args2.odesc.desc(),
                  output_grad_data + i * group_offset_out, args2.cdesc.desc(),
                  filter_algo, cudnn_workspace_ptr, workspace_size, &beta,
                  args2.wdesc.desc(),
                  filter_grad_data + i * group_offset_filter));
            },
            workspace_size);
      }
    }
  }
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
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      "It must use CUDAPlace.");
    auto X = ctx.Input<Tensor>("Input");
    auto W = ctx.Input<Tensor>("Filter");
    auto dO = ctx.Input<Tensor>("DOutput");
    auto ddX = ctx.Input<Tensor>("DDInput");
    auto ddW = ctx.Input<Tensor>("DDFilter");

    auto ddO = ctx.Output<Tensor>("DDOutput");
    auto dW = ctx.Output<Tensor>("DFilter");
    auto dX = ctx.Output<Tensor>("DInput");
    if (ddO) {
      ddO->mutable_data<T>(ctx.GetPlace());
    }
    if (dW) {
      dW->mutable_data<T>(ctx.GetPlace());
    }
    if (dX) {
      dX->mutable_data<T>(ctx.GetPlace());
    }

    // const T* x = X->data<T>();
    const T* dy = dO->data<T>();
    const T* w = W->data<T>();

    const T* ddx = nullptr;
    const T* ddw = nullptr;
    T *dw, *dx, *ddy;
    dw = dx = ddy = nullptr;
    T* transformed_dx = nullptr;
    const std::vector<int>& strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    bool exhaustive_search =
        FLAGS_cudnn_exhaustive_search || ctx.Attr<bool>("exhaustive_search");
    bool deterministic = FLAGS_cudnn_deterministic;
    if (exhaustive_search && deterministic) {
      PADDLE_THROW(
          "Can't set exhaustive_search True and "
          "FLAGS_cudnn_deterministic True at same time.");
    }
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");

    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // transform Tensors to channel first-----------
    Tensor transformed_X_channel(X->type());
    Tensor transformed_dO_channel(dO->type());
    Tensor transformed_ddX_channel(ddX->type());

    Tensor transformed_ddO_channel(dO->type());
    Tensor transformed_dX_channel(X->type());

    if (channel_last) {
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, X, &transformed_X_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, X, &transformed_X_channel);

      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, dO, &transformed_dO_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, dO, &transformed_dO_channel);

      ResizeToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, ddX, &transformed_ddX_channel);
      TransToChannelFirst<platform::CUDADeviceContext, T>(
          ctx, ddX, &transformed_ddX_channel);

      if (ddO) {
        ResizeToChannelFirst<platform::CUDADeviceContext, T>(
            ctx, ddO, &transformed_ddO_channel);
      }
      if (dX) {
        ResizeToChannelFirst<platform::CUDADeviceContext, T>(
            ctx, dX, &transformed_dX_channel);
        transformed_dX_channel.mutable_data<T>(ctx.GetPlace());
      }

    } else {
      transformed_X_channel = *X;
      transformed_dO_channel = *dO;
      transformed_ddX_channel = *ddX;
      if (ddO) {
        transformed_ddO_channel.ShareDataWith(*ddO);
      }
      if (dX) {
        transformed_dX_channel.ShareDataWith(*dX);
      }
    }

    auto in_dims = transformed_X_channel.dims();
    auto filter_dims = W->dims();
    framework::DDim in_data_dims =
        framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    int data_dim = strides.size();  // 2d or 3d
    bool is_sys_pad = IsSymmetricPadding(paddings, data_dim);
    Tensor transformed_X(X->type());
    Tensor transformed_ddX(X->type());

    Tensor transformed_dX(X->type());

    std::vector<int> padding_common(data_dim, 0);
    std::vector<int> input_pad(X->dims().size() * 2, 0);

    if (!is_sys_pad) {
      // get pad
      std::vector<int> padding_diff(data_dim);
      std::vector<int> new_input_shape_vec(data_dim + 2);
      new_input_shape_vec[0] = transformed_X_channel.dims()[0];
      new_input_shape_vec[1] = transformed_X_channel.dims()[1];

      for (size_t i = 0; i < data_dim; ++i) {
        padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
        padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
        new_input_shape_vec[i + 2] =
            transformed_X_channel.dims()[i + 2] + padding_diff[i];
        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
      framework::DDim new_input_shape(
          framework::make_ddim(new_input_shape_vec));
      transformed_X.Resize(new_input_shape);
      transformed_ddX.Resize(new_input_shape);
      transformed_dX.Resize(new_input_shape);
      auto& dev_ctx =
          ctx.template device_context<paddle::platform::CUDADeviceContext>();

      transformed_X =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_input_shape, dev_ctx);
      transformed_ddX =
          ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
              new_input_shape, dev_ctx);
      if (dX) {
        transformed_dX =
            ctx.AllocateTmpTensor<T, paddle::platform::CUDADeviceContext>(
                new_input_shape, dev_ctx);
      }

      // pad for input
      const int rank = X->dims().size();
      T pad_value(0.0);
      switch (rank) {
        case 4: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, transformed_X_channel, pad_value, &transformed_X);
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 4>(
              ctx, input_pad, transformed_ddX_channel, pad_value,
              &transformed_ddX);
        } break;
        case 5: {
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, transformed_X_channel, pad_value, &transformed_X);
          math::PadFunction<paddle::platform::CUDADeviceContext, T, 5>(
              ctx, input_pad, transformed_ddX_channel, pad_value,
              &transformed_ddX);
        } break;
        default:
          PADDLE_THROW("ConvOp only support tensors with 4 or 5 dimensions.");
      }

    } else {
      transformed_X.ShareDataWith(transformed_X_channel);
      transformed_ddX.ShareDataWith(transformed_ddX_channel);
      if (dX) {
        transformed_dX.ShareDataWith(transformed_dX_channel);
      }

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

    const T* x = transformed_X.data<T>();

    int iwo_group = groups;
    int c_group = 1;
#if CUDNN_VERSION_MIN(7, 0, 1)
    iwo_group = 1;
    c_group = groups;
#endif
    auto dtype = platform::CudnnDataType<T>::type;

    auto handle = dev_ctx.cudnn_handle();

    ConvArgs args1{&transformed_ddX,         W,
                   &transformed_ddO_channel, strides,
                   padding_common,           dilations};
    ConvArgs args2{&transformed_X, ddW,      &transformed_ddO_channel, strides,
                   padding_common, dilations};
    ConvArgs args3{&transformed_ddX, dW,       &transformed_dO_channel, strides,
                   padding_common,   dilations};
    ConvArgs args4{&transformed_dX, ddW,      &transformed_dO_channel, strides,
                   padding_common,  dilations};

    cudnnConvolutionFwdAlgo_t fwd_algo1 =
        static_cast<cudnnConvolutionFwdAlgo_t>(0);
    cudnnConvolutionFwdAlgo_t fwd_algo2 =
        static_cast<cudnnConvolutionFwdAlgo_t>(0);
    cudnnConvolutionBwdDataAlgo_t data_algo =
        static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
    cudnnConvolutionBwdFilterAlgo_t filter_algo =
        static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);

    auto layout = GetCudnnTensorFormat(DataLayout::kNCHW);

    // ddo = conv(ddI, W) + conv(I, ddW)
    size_t workspace_size = 0;

    T* transformed_ddy_channel = nullptr;
    if (ddO) {
      ddy = ddO->data<T>();
      transformed_ddy_channel = transformed_ddO_channel.data<T>();
      if (ddX) {
        args1.handle = handle;
        args1.idesc.set(transformed_ddX, iwo_group);
        args1.wdesc.set(*W, layout, iwo_group);
        args1.odesc.set(transformed_ddO_channel, iwo_group);
        args1.cdesc.set(dtype, padding_common, strides, dilations, c_group);

        using search1 = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
        fwd_algo1 = search1::Find<T>(args1, exhaustive_search, false, 0, ctx);
        workspace_size = search1::GetWorkspaceSize(args1, fwd_algo1);
      }

      if (ddW) {
        ddw = ddW->data<T>();
        args2.handle = handle;
        args2.idesc.set(transformed_X, iwo_group);

        args2.wdesc.set(*ddW, layout, iwo_group);

        args2.odesc.set(transformed_ddO_channel, iwo_group);
        args2.cdesc.set(dtype, padding_common, strides, dilations, c_group);

        using search2 = SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
        fwd_algo2 = search2::Find<T>(args2, exhaustive_search, false, 0, ctx);
        workspace_size = std::max(workspace_size,
                                  search2::GetWorkspaceSize(args2, fwd_algo2));
      }
    }

    if (dW && ddX) {
      dw = dW->data<T>();
      args3.handle = handle;
      args3.idesc.set(transformed_ddX, iwo_group);
      args3.wdesc.set(*dW, layout, iwo_group);

      args3.odesc.set(transformed_dO_channel, iwo_group);

      args3.cdesc.set(dtype, padding_common, strides, dilations, c_group);

      using search3 = SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
      filter_algo =
          search3::Find<T>(args3, exhaustive_search, deterministic, 1, ctx);
      workspace_size = std::max(workspace_size,
                                search3::GetWorkspaceSize(args3, filter_algo));
    }

    if (ddW && dX) {
      transformed_dx = transformed_dX.data<T>();

      args4.handle = handle;
      args4.idesc.set(transformed_dX, iwo_group);
      args4.wdesc.set(*ddW, layout, iwo_group);
      args4.odesc.set(transformed_dO_channel, iwo_group);
      args4.cdesc.set(dtype, padding_common, strides, dilations, c_group);

      using search4 = SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
      data_algo =
          search4::Find<T>(args4, exhaustive_search, deterministic, 2, ctx);
      workspace_size =
          std::max(workspace_size, search4::GetWorkspaceSize(args4, data_algo));
    }

    int i_n, i_c, i_d, i_h, i_w;
    GetNCDHW(transformed_X.dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d, &i_h,
             &i_w);

    int o_n, o_c, o_d, o_h, o_w;
    GetNCDHW(transformed_dO_channel.dims(), DataLayout::kNCHW, &o_n, &o_c, &o_d,
             &o_h, &o_w);

    int group_offset_in = i_c / groups * i_h * i_w * i_d;
    int group_offset_out = o_c / groups * o_h * o_w * o_d;
    int group_offset_filter = W->numel() / groups;

    ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
    auto wkspace_handle = dev_ctx.cudnn_workspace_handle();

    if (ddO) {
      if (ddX) {
        ddx = transformed_ddX.data<T>();
        for (int i = 0; i < groups; i++) {
          wkspace_handle.RunFunc(
              [&](void* workspace_ptr) {
                CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
                    handle, &alpha, args1.idesc.desc(),
                    ddx + i * group_offset_in, args1.wdesc.desc(),
                    w + i * group_offset_filter, args1.cdesc.desc(), fwd_algo1,
                    workspace_ptr, workspace_size, &beta, args1.odesc.desc(),
                    transformed_ddy_channel + i * group_offset_out));
              },
              workspace_size);
        }
      }
      if (ddW) {
        for (int i = 0; i < groups; i++) {
          wkspace_handle.RunFunc(
              [&](void* workspace_ptr) {
                CUDNN_ENFORCE(platform::dynload::cudnnConvolutionForward(
                    handle, &alpha, args2.idesc.desc(), x + i * group_offset_in,
                    args2.wdesc.desc(), ddw + i * group_offset_filter,
                    args2.cdesc.desc(), fwd_algo2, workspace_ptr,
                    workspace_size, &alpha, args2.odesc.desc(),
                    transformed_ddy_channel + i * group_offset_out));
              },
              workspace_size);
        }
      }
      if (channel_last) {
        TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
            ctx, &transformed_ddO_channel, ddO);
      }
    }
    T* transformed_dy_channel = nullptr;
    if (dW && ddX) {
      ddx = transformed_ddX.data<T>();
      transformed_dy_channel = transformed_dO_channel.data<T>();
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardFilter(
                  handle, &alpha, args3.idesc.desc(), ddx + i * group_offset_in,
                  args3.odesc.desc(),
                  transformed_dy_channel + i * group_offset_out,
                  args3.cdesc.desc(), filter_algo, workspace_ptr,
                  workspace_size, &beta, args3.wdesc.desc(),
                  dw + i * group_offset_filter));
            },
            workspace_size);
      }
    }

    if (dX && ddW) {
      ddw = ddW->data<T>();
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
                  handle, &alpha, args4.wdesc.desc(),
                  ddw + i * group_offset_filter, args4.odesc.desc(),
                  transformed_dy_channel + i * group_offset_out,
                  args4.cdesc.desc(), data_algo, workspace_ptr, workspace_size,
                  &beta, args4.idesc.desc(),
                  transformed_dx + i * group_offset_in));
            },
            workspace_size);
      }

      // reverse padded input
      std::vector<int> starts(X->dims().size(), 0);
      std::vector<int> axes(X->dims().size(), 0);

      for (size_t i = 0; i < X->dims().size(); ++i) {
        starts[i] = input_pad[2 * i];
        axes[i] = i;
      }
      if (X->dims().size() == 4) {
        Slice_2<paddle::platform::CUDADeviceContext, T, 4>(
            ctx, &transformed_dX, &transformed_dX_channel, starts, axes);
      } else {
        Slice_2<paddle::platform::CUDADeviceContext, T, 5>(
            ctx, &transformed_dX, &transformed_dX_channel, starts, axes);
      }
      if (channel_last) {
        TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
            ctx, &transformed_dX_channel, dX);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
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
