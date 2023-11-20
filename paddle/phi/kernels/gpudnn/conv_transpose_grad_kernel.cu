/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/conv_transpose_grad_kernel.h"

#include <algorithm>

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/padding.h"
#include "paddle/phi/kernels/funcs/slice.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/gpu/rocm/miopen_helper.h"
#include "paddle/phi/kernels/gpudnn/conv_miopen_helper.h"
#else
#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_v7.h"
#endif

namespace phi {

using GPUDNNDataLayout = phi::backends::gpu::DataLayout;

template <typename T, typename Context>
void ConvTransposeGradRawGPUDNNKernel(const Context& ctx,
                                      const DenseTensor& x,
                                      const DenseTensor& filter,
                                      const DenseTensor& dout,
                                      const std::vector<int>& strides,
                                      const std::vector<int>& paddings,
                                      const std::string& padding_algorithm,
                                      int groups,
                                      const std::vector<int>& dilations,
                                      const std::string& data_format,
                                      DenseTensor* dx,
                                      DenseTensor* dfilter) {
  const T* filter_data = filter.data<T>();
  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ =
      dilations;  // cudnn v5 does not support dilations
  const GPUDNNDataLayout data_layout =
      (data_format != "NHWC" ? GPUDNNDataLayout::kNCHW
                             : GPUDNNDataLayout::kNHWC);

  // if channel_last, transpose to channel_first
  DenseTensor x_transpose;
  DenseTensor dout_transpose;
  std::vector<int> x_vec = common::vectorize<int>(x.dims());
  std::vector<int> out_vec = common::vectorize<int>(dout.dims());
  if (data_layout == GPUDNNDataLayout::kNHWC) {
    if (strides.size() == 2U) {
      std::vector<int> axis = {0, 3, 1, 2};
      for (size_t i = 0; i < axis.size(); ++i) {
        x_vec[i] = x.dims()[axis[i]];
        out_vec[i] = dout.dims()[axis[i]];
      }
      x_transpose = Transpose<T, Context>(ctx, x, axis);
      dout_transpose = Transpose<T, Context>(ctx, dout, axis);
    } else if (strides.size() == 3U) {
      std::vector<int> axis = {0, 4, 1, 2, 3};
      for (size_t i = 0; i < axis.size(); ++i) {
        x_vec[i] = x.dims()[axis[i]];
        out_vec[i] = dout.dims()[axis[i]];
      }
      x_transpose = Transpose<T, Context>(ctx, x, axis);
      dout_transpose = Transpose<T, Context>(ctx, dout, axis);
    }
  } else {
    x_transpose = x;
    dout_transpose = dout;
  }

  // update padding and dilation
  auto x_dims = x_transpose.dims();
  auto filter_dims = filter.dims();
  DDim x_data_dims;
  x_data_dims = slice_ddim(x_dims, 2, x_dims.size());
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, x_data_dims, strides, ksize);

  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings_, data_dim);

  std::vector<int> x_pad(x_dims.size() * 2, 0);
  DenseTensor transformed_dout;
  std::vector<int> padding_common(data_dim, 0);
  if (!is_sys_pad) {
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_dout_shape_vec(data_dim + 2);
    new_dout_shape_vec[0] = dout_transpose.dims()[0];
    new_dout_shape_vec[1] = dout_transpose.dims()[1];

    for (size_t i = 0; i < data_dim; ++i) {
      padding_diff[i] = std::abs(paddings_[2 * i] - paddings_[2 * i + 1]);
      padding_common[i] = std::min(paddings_[2 * i], paddings_[2 * i + 1]);
      new_dout_shape_vec[i + 2] =
          dout_transpose.dims()[i + 2] + padding_diff[i];
      x_pad[2 * i + 4] = paddings_[2 * i] - padding_common[i];
      x_pad[2 * i + 4 + 1] = paddings_[2 * i + 1] - padding_common[i];
    }

    transformed_dout.Resize(common::make_ddim(new_dout_shape_vec));
    ctx.template Alloc<T>(&transformed_dout);

    const int rank = x_transpose.dims().size();
    T pad_value(0.0);
    switch (rank) {
      case 4: {
        funcs::PadFunction<Context, T, 4>(
            ctx, x_pad, dout_transpose, pad_value, &transformed_dout);
      } break;
      case 5: {
        funcs::PadFunction<Context, T, 5>(
            ctx, x_pad, dout_transpose, pad_value, &transformed_dout);
      } break;
      default:
        PADDLE_THROW(errors::InvalidArgument(
            "Op(ConvTranspose) only supports 4-D or 5-D x DenseTensor."));
    }
  } else {
    transformed_dout = dout_transpose;
    if (paddings_.size() == data_dim) {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings_[i];
      }
    } else {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings_[2 * i];
      }
    }
  }

  const T* x_data = x_transpose.data<T>();
  const T* dout_data = transformed_dout.data<T>();
  out_vec = common::vectorize<int>(transformed_dout.dims());

  // ------------------- cudnn descriptors ---------------------
  GPUDNNDataLayout layout;

  if (strides.size() == 2U) {
    layout = GPUDNNDataLayout::kNCHW;
  } else {
    layout = GPUDNNDataLayout::kNCDHW;
  }

  int iwo_groups = groups;
  int c_groups = 1;
#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
  iwo_groups = 1;
  c_groups = groups;
  groups = 1;
#endif

  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;
  auto handle = ctx.cudnn_handle();

  ConvArgs args1{handle,
                 &transformed_dout,
                 &filter,
                 &x_transpose,
                 strides,
                 padding_common,
                 dilations_,
                 dtype,
                 groups,
                 layout};
  ConvArgs args2{handle,
                 &transformed_dout,
                 &filter,
                 &x_transpose,
                 strides,
                 padding_common,
                 dilations_,
                 dtype,
                 groups,
                 layout};

#ifdef PADDLE_WITH_HIP
  SearchResult<miopenConvFwdAlgorithm_t> fwd_result;
  SearchResult<miopenConvBwdWeightsAlgorithm_t> filter_result;
#else
  SearchResult<cudnnConvolutionFwdAlgo_t> fwd_result;
  SearchResult<cudnnConvolutionBwdFilterAlgo_t> filter_result;
#endif

  auto layout_tensor = phi::backends::gpu::GetCudnnTensorFormat(layout);
  size_t workspace_size = 0;
  bool deterministic = FLAGS_cudnn_deterministic;
  T* dx_data = nullptr;
  T* dfilter_data = nullptr;

  if (dx) {
    dx_data = ctx.template Alloc<T>(dx);

    args1.idesc.set(transformed_dout, iwo_groups);
    args1.wdesc.set(filter, layout_tensor, iwo_groups);
    args1.odesc.set(x_transpose, iwo_groups);
    args1.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations_,
                    phi::AllowTF32Cudnn(),
                    c_groups);
#ifdef PADDLE_WITH_HIP
    using search1 = SearchAlgorithm<miopenConvFwdAlgorithm_t>;
    workspace_size = std::max(workspace_size, search1::GetWorkspaceSize(args1));
    fwd_result.algo =
        search1::Find<T>(args1, false, deterministic, workspace_size, ctx);
#else
    using search1 = SearchAlgorithm<ConvKind::kForward>;
    fwd_result = search1::Find<T>(ctx, args1, false, deterministic, false);
    workspace_size = std::max(
        workspace_size, search1::GetWorkspaceSize(args1, fwd_result.algo));
#endif
  }

  if (dfilter) {
    dfilter_data = ctx.template Alloc<T>(dfilter);

    args2.idesc.set(transformed_dout, iwo_groups);
    args2.wdesc.set(*dfilter, layout_tensor, iwo_groups);
    args2.odesc.set(x_transpose, iwo_groups);
    args2.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations_,
                    phi::AllowTF32Cudnn(),
                    c_groups);
#ifdef PADDLE_WITH_HIP
    using search2 = SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t>;
    workspace_size = std::max(workspace_size, search2::GetWorkspaceSize(args2));
    filter_result.algo =
        search2::Find<T>(args2, false, deterministic, workspace_size, ctx);
#else
    using search2 = SearchAlgorithm<ConvKind::kBackwardFilter>;
    filter_result = search2::Find<T>(ctx, args2, false, deterministic, false);
    workspace_size = std::max(
        workspace_size, search2::GetWorkspaceSize(args2, filter_result.algo));
#endif
  }

  // ------------------- cudnn conv backward data ---------------------
  // FIxME(typhoonzero): template type T may not be the same as cudnn call.
  int x_offset = x.numel() / x.dims()[0] / groups;
  int dout_offset =
      transformed_dout.numel() / transformed_dout.dims()[0] / groups;
  int filter_offset = filter.numel() / groups;
  ScalingParamType<T> alpha = 1.0f;
  ScalingParamType<T> beta = 0.0f;
  auto workspace_handle = ctx.cudnn_workspace_handle();
  if (dx) {
#ifdef PADDLE_WITH_HIP
    // Because beta is zero, it is unnecessary to reset dx.
    for (int g = 0; g < groups; g++) {
      auto cudnn_func = [&](void* cudnn_workspace) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            dynload::miopenConvolutionForward(handle,
                                              &alpha,
                                              args1.idesc.desc(),
                                              dout_data + dout_offset * g,
                                              args1.wdesc.desc(),
                                              filter_data + filter_offset * g,
                                              args1.cdesc.desc(),
                                              fwd_result.algo,
                                              &beta,
                                              args1.odesc.desc(),
                                              dx_data + x_offset * g,
                                              cudnn_workspace,
                                              workspace_size));
      };
      workspace_handle.RunFunc(cudnn_func, workspace_size);
    }
#else   // PADDLE_WITH_HIP
    ConvRunner<T, ConvKind::kForward>::Apply(ctx,
                                             args1,
                                             fwd_result,
                                             dout_data,
                                             filter_data,
                                             dx_data,
                                             groups,
                                             dout_offset,
                                             filter_offset,
                                             x_offset,
                                             workspace_size,
                                             &workspace_handle,
                                             false);
#endif  // PADDLE_WITH_HIP

    if (data_layout == GPUDNNDataLayout::kNHWC) {
      DenseTensor dx_transpose;
      DenseTensor dx_nchw;
      dx_nchw.ShareDataWith(*dx);
      dx_nchw.Resize(common::make_ddim(x_vec));
      if (strides.size() == 2U) {
        std::vector<int> axis = {0, 2, 3, 1};
        dx_transpose = Transpose<T, Context>(ctx, dx_nchw, axis);
        *dx = dx_transpose;
      } else if (strides.size() == 3U) {
        std::vector<int> axis = {0, 2, 3, 4, 1};
        dx_transpose = Transpose<T, Context>(ctx, dx_nchw, axis);
        *dx = dx_transpose;
      }
    }
  }

  // ------------------- cudnn conv backward filter ---------------------
  if (dfilter) {
    // Because beta is zero, it is unnecessary to reset dfilter.
    // Gradient with respect to the filter
#ifdef PADDLE_WITH_HIP
    for (int g = 0; g < groups; g++) {
      auto cudnn_func = [&](void* cudnn_workspace) {
        PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenConvolutionBackwardWeights(
            handle,
            &alpha,
            args2.odesc.desc(),
            x_data + x_offset * g,
            args2.idesc.desc(),
            dout_data + dout_offset * g,
            args2.cdesc.desc(),
            filter_result.algo,
            &beta,
            args2.wdesc.desc(),
            dfilter_data + filter_offset * g,
            cudnn_workspace,
            workspace_size));
      };
      workspace_handle.RunFunc(cudnn_func, workspace_size);
    }
#else   // PADDLE_WITH_HIP
    ConvRunner<T, ConvKind::kBackwardFilter>::Apply(ctx,
                                                    args2,
                                                    filter_result,
                                                    x_data,
                                                    dout_data,
                                                    dfilter_data,
                                                    groups,
                                                    dout_offset,
                                                    filter_offset,
                                                    x_offset,
                                                    workspace_size,
                                                    &workspace_handle,
                                                    false);
#endif  // PADDLE_WITH_HIP
  }
}

template <typename T, typename Context>
void Conv2dTransposeGradGPUDNNKernel(const Context& ctx,
                                     const DenseTensor& x,
                                     const DenseTensor& filter,
                                     const DenseTensor& dout,
                                     const std::vector<int>& strides,
                                     const std::vector<int>& paddings_,
                                     const std::vector<int>& output_padding,
                                     const IntArray& output_size,
                                     const std::string& padding_algorithm,
                                     int groups,
                                     const std::vector<int>& dilations_,
                                     const std::string& data_format,
                                     DenseTensor* dx,
                                     DenseTensor* dfilter) {
  ConvTransposeGradRawGPUDNNKernel<T, Context>(ctx,
                                               x,
                                               filter,
                                               dout,
                                               strides,
                                               paddings_,
                                               padding_algorithm,
                                               groups,
                                               dilations_,
                                               data_format,
                                               dx,
                                               dfilter);
}

/*
 * Inputs:  I, filter, dout, ddI, ddfilter
 * Outputs: ddout, dfilter, dI
 * ddo = conv_bp_data(filter, ddI) + conv_bp_data(ddfilter, I)
 * dfilter = conv_bp_filter(dout, ddI)
 * dI = conv(dout, ddfilter)
 */
template <typename T, typename Context>
void Conv2dTransposeDoubleGradGPUDNNKernel(
    const Context& ctx,
    const DenseTensor& x,
    const DenseTensor& filter,
    const DenseTensor& dout,
    const DenseTensor& ddx,
    const DenseTensor& ddfilter,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& output_padding,
    const IntArray& output_size,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations,
    const std::string& data_format,
    DenseTensor* dx,
    DenseTensor* dfilter,
    DenseTensor* ddout) {
  if (dx) {
    ctx.template Alloc<T>(dx);
  }
  if (dfilter) {
    ctx.template Alloc<T>(dfilter);
  }
  if (ddout) {
    ctx.template Alloc<T>(ddout);
    funcs::SetConstant<Context, T> set_zero;
    set_zero(ctx, ddout, static_cast<T>(0));
  }

  const T* filter_ = filter.data<T>();
  const T* dout_ = dout.data<T>();
  const T* ddx_ = nullptr;
  const T* ddfilter_ = nullptr;
  T* dx_ = nullptr;
  T* dfilter_ = nullptr;
  T* ddout_ = nullptr;
  T* transformed_dx_ = nullptr;

  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;

  bool deterministic = FLAGS_cudnn_deterministic;
  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  // transform DenseTensors to channel first-----------
  DenseTensor transformed_x_channel(x.type());
  DenseTensor transformed_dout_channel(dout.type());
  DenseTensor transformed_ddx_channel(x.type());

  DenseTensor transformed_dx_channel(x.type());
  DenseTensor transformed_ddout_channel(dout.type());

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(ctx, &x, &transformed_x_channel);
    TransToChannelFirst<Context, T>(ctx, &x, &transformed_x_channel);

    ResizeToChannelFirst<Context, T>(ctx, &dout, &transformed_dout_channel);
    TransToChannelFirst<Context, T>(ctx, &dout, &transformed_dout_channel);

    ResizeToChannelFirst<Context, T>(ctx, &ddx, &transformed_ddx_channel);
    TransToChannelFirst<Context, T>(ctx, &ddx, &transformed_ddx_channel);

    if (dx) {
      ResizeToChannelFirst<Context, T>(ctx, dx, &transformed_dx_channel);
      ctx.template Alloc<T>(&transformed_dx_channel);
    }
    if (ddout) {
      ResizeToChannelFirst<Context, T>(ctx, ddout, &transformed_ddout_channel);
    }
  } else {
    transformed_x_channel = x;
    transformed_dout_channel = dout;
    transformed_ddx_channel = ddx;

    if (dx) {
      transformed_dx_channel = *dx;
    }
  }
  std::vector<int> out_vec =
      common::vectorize<int>(transformed_dout_channel.dims());

  auto x_dims = transformed_x_channel.dims();
  auto filter_dims = filter.dims();
  DDim x_data_dims = slice_ddim(x_dims, 2, x_dims.size());
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, x_data_dims, strides, ksize);

  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings_, data_dim);
  DenseTensor transformed_x(x.type());
  DenseTensor transformed_ddx(x.type());

  DenseTensor transformed_dout(dout.type());

  std::vector<int> padding_common(data_dim, 0);
  std::vector<int> input_pad(x.dims().size() * 2, 0);

  if (!is_sys_pad) {
    // get pad
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_input_shape_vec(data_dim + 2);
    std::vector<int> new_output_grad_shape_vec(data_dim + 2);

    new_input_shape_vec[0] = transformed_x_channel.dims()[0];
    new_input_shape_vec[1] = transformed_x_channel.dims()[1];

    new_output_grad_shape_vec[0] = transformed_dout_channel.dims()[0];
    new_output_grad_shape_vec[1] = transformed_dout_channel.dims()[1];

    for (size_t i = 0; i < data_dim; ++i) {
      padding_diff[i] = std::abs(paddings_[2 * i] - paddings_[2 * i + 1]);
      padding_common[i] = std::min(paddings_[2 * i], paddings_[2 * i + 1]);
      new_input_shape_vec[i + 2] =
          transformed_x_channel.dims()[i + 2] + padding_diff[i];

      new_output_grad_shape_vec[i + 2] =
          transformed_dout_channel.dims()[i + 2] + padding_diff[i];

      input_pad[2 * i + 4] = paddings_[2 * i] - padding_common[i];
      input_pad[2 * i + 4 + 1] = paddings_[2 * i + 1] - padding_common[i];
    }
    DDim new_input_shape(common::make_ddim(new_input_shape_vec));
    transformed_x.Resize(new_input_shape);
    transformed_ddx.Resize(new_input_shape);
    transformed_dout.Resize(common::make_ddim(new_output_grad_shape_vec));

    ctx.template Alloc<T>(&transformed_x);
    ctx.template Alloc<T>(&transformed_ddx);
    ctx.template Alloc<T>(&transformed_dout);

    // pad for input
    const int rank = x.dims().size();
    T pad_value(0.0);
    switch (rank) {
      case 4: {
        funcs::PadFunction<Context, T, 4>(
            ctx, input_pad, transformed_x_channel, pad_value, &transformed_x);
        funcs::PadFunction<Context, T, 4>(ctx,
                                          input_pad,
                                          transformed_dout_channel,
                                          pad_value,
                                          &transformed_dout);
        funcs::PadFunction<Context, T, 4>(ctx,
                                          input_pad,
                                          transformed_ddx_channel,
                                          pad_value,
                                          &transformed_ddx);
      } break;
      case 5: {
        funcs::PadFunction<Context, T, 5>(
            ctx, input_pad, transformed_x_channel, pad_value, &transformed_x);
        funcs::PadFunction<Context, T, 5>(ctx,
                                          input_pad,
                                          transformed_ddx_channel,
                                          pad_value,
                                          &transformed_ddx);
      } break;
      default:
        PADDLE_THROW(errors::InvalidArgument(
            "ConvOp only support tensors with 4 or 5 dimensions."));
    }
  } else {
    transformed_x = transformed_x_channel;
    transformed_dout = transformed_dout_channel;
    transformed_ddx = transformed_ddx_channel;

    if (paddings_.size() == data_dim) {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings_[i];
      }
    } else {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings_[2 * i];
      }
    }
  }

  std::vector<int64_t> starts(data_dim, 0);
  std::vector<int64_t> ends(data_dim, 0);
  std::vector<int64_t> axes(data_dim, 0);
  for (size_t i = 0; i < data_dim; ++i) {
    starts[i] = input_pad[2 * i + 4] * (strides[i] + 1);
    ends[i] = starts[i] + out_vec[i + 2];
    axes[i] = i + 2;
  }

  std::vector<int> transformed_out_vec = out_vec;
  for (size_t i = 0; i < data_dim; ++i) {
    transformed_out_vec[i + 2] =
        out_vec[i + 2] +
        (input_pad[2 * i + 4] + input_pad[2 * i + 5]) * strides[i] -
        2 * padding_common[i] + paddings_[2 * i] + paddings_[2 * i + 1];
  }

  if (!is_sys_pad) {
    transformed_ddout_channel.Resize(common::make_ddim(transformed_out_vec));
    ctx.template Alloc<T>(&transformed_ddout_channel);
  } else {
    ctx.template Alloc<T>(ddout);
    transformed_ddout_channel = *ddout;
    transformed_ddout_channel.Resize(common::make_ddim(transformed_out_vec));
  }

  const T* x_ = transformed_x.data<T>();

  int iwo_group = groups;
  int c_group = 1;
#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
  iwo_group = 1;
  c_group = groups;
  groups = 1;
#endif
  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;

  auto handle = ctx.cudnn_handle();
  auto layout =
      phi::backends::gpu::GetCudnnTensorFormat(GPUDNNDataLayout::kNCHW);

  ConvArgs args1{handle,
                 &transformed_ddout_channel,
                 &filter,
                 &transformed_ddx,
                 strides,
                 padding_common,
                 dilations_,
                 dtype,
                 groups,
                 GPUDNNDataLayout::kNCHW};
  ConvArgs args2{handle,
                 &transformed_ddout_channel,
                 &ddfilter,
                 &transformed_x,
                 strides,
                 padding_common,
                 dilations_,
                 dtype,
                 groups,
                 GPUDNNDataLayout::kNCHW};

  ConvArgs args3{handle,
                 &transformed_dout,
                 dfilter,
                 &transformed_ddx_channel,
                 strides,
                 padding_common,
                 dilations_,
                 dtype,
                 groups,
                 GPUDNNDataLayout::kNCHW};
  ConvArgs args4{handle,
                 &transformed_dout,
                 &ddfilter,
                 &transformed_dx_channel,
                 strides,
                 padding_common,
                 dilations_,
                 dtype,
                 groups,
                 GPUDNNDataLayout::kNCHW};
#ifdef PADDLE_WITH_HIP
  SearchResult<miopenConvBwdDataAlgorithm_t> bwd_result1;
  SearchResult<miopenConvBwdDataAlgorithm_t> bwd_result2;
  SearchResult<miopenConvBwdWeightsAlgorithm_t> filter_result;
  SearchResult<miopenConvFwdAlgorithm_t> fwd_result;
#else
  SearchResult<cudnnConvolutionBwdDataAlgo_t> bwd_result1;
  SearchResult<cudnnConvolutionBwdDataAlgo_t> bwd_result2;
  SearchResult<cudnnConvolutionBwdFilterAlgo_t> filter_result;
  SearchResult<cudnnConvolutionFwdAlgo_t> fwd_result;
#endif

  // ddo = conv(ddI, filter) + conv(I, ddfilter)
  size_t workspace_size = 0;

  T* transformed_ddout_channel_ = nullptr;

  if (ddout) {
    ddout_ = ddout->data<T>();
    transformed_ddout_channel_ = transformed_ddout_channel.data<T>();

    args1.idesc.set(transformed_ddout_channel, iwo_group);
    args1.wdesc.set(filter, layout, iwo_group);
    args1.odesc.set(transformed_ddx, iwo_group);
    args1.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations_,
                    phi::AllowTF32Cudnn(),
                    c_group);
#ifdef PADDLE_WITH_HIP
    using search1 = SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
    workspace_size = search1::GetWorkspaceSize(args1);
    bwd_result1.algo =
        search1::Find<T>(args1, false, deterministic, workspace_size, ctx);
#else
    using search1 = SearchAlgorithm<ConvKind::kBackwardData>;
    bwd_result1 = search1::Find<T>(ctx, args1, false, deterministic, false);
    workspace_size = search1::GetWorkspaceSize(args1, bwd_result1.algo);
#endif

    ddfilter_ = ddfilter.data<T>();
    args2.handle = handle;
    args2.idesc.set(transformed_ddout_channel, iwo_group);
    args2.wdesc.set(ddfilter, layout, iwo_group);
    args2.odesc.set(transformed_x, iwo_group);
    args2.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations_,
                    phi::AllowTF32Cudnn(),
                    c_group);
#ifdef PADDLE_WITH_HIP
    using search2 = SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
    workspace_size = std::max(workspace_size, search2::GetWorkspaceSize(args2));
    bwd_result2.algo =
        search2::Find<T>(args2, false, deterministic, workspace_size, ctx);
#else
    using search2 = SearchAlgorithm<ConvKind::kBackwardData>;
    bwd_result2 = search2::Find<T>(ctx, args2, false, deterministic, false);
    workspace_size = std::max(
        workspace_size, search2::GetWorkspaceSize(args2, bwd_result2.algo));
#endif
  }

  if (dfilter) {
    dfilter_ = dfilter->data<T>();

    args3.idesc.set(transformed_dout, iwo_group);
    args3.wdesc.set(*dfilter, layout, iwo_group);
    args3.odesc.set(transformed_ddx_channel, iwo_group);
    args3.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations_,
                    phi::AllowTF32Cudnn(),
                    c_group);
#ifdef PADDLE_WITH_HIP
    using search3 = SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t>;
    workspace_size = std::max(workspace_size, search3::GetWorkspaceSize(args3));
    filter_result.algo =
        search3::Find<T>(args3, false, deterministic, workspace_size, ctx);
#else
    using search3 = SearchAlgorithm<ConvKind::kBackwardFilter>;
    filter_result = search3::Find<T>(ctx, args3, false, deterministic, false);
    workspace_size = std::max(
        workspace_size, search3::GetWorkspaceSize(args3, filter_result.algo));
#endif
  }

  if (dx) {
    transformed_dx_ = transformed_dx_channel.data<T>();

    args4.handle = handle;
    args4.idesc.set(transformed_dout, iwo_group);
    args4.wdesc.set(ddfilter, layout, iwo_group);
    args4.odesc.set(transformed_dx_channel, iwo_group);
    args4.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations_,
                    phi::AllowTF32Cudnn(),
                    c_group);
#ifdef PADDLE_WITH_HIP
    using search4 = SearchAlgorithm<miopenConvFwdAlgorithm_t>;
    workspace_size = std::max(workspace_size, search4::GetWorkspaceSize(args4));
    fwd_result.algo =
        search4::Find<T>(args4, false, deterministic, workspace_size, ctx);
#else
    using search4 = SearchAlgorithm<ConvKind::kForward>;
    fwd_result = search4::Find<T>(ctx, args4, false, deterministic, false);
    workspace_size = std::max(
        workspace_size, search4::GetWorkspaceSize(args4, fwd_result.algo));
#endif
  }

  int i_n, i_c, i_d, i_h, i_w;
  GetNCDHW(transformed_x.dims(),
           GPUDNNDataLayout::kNCHW,
           &i_n,
           &i_c,
           &i_d,
           &i_h,
           &i_w);

  int o_n, o_c, o_d, o_h, o_w;
  GetNCDHW(transformed_dout.dims(),
           GPUDNNDataLayout::kNCHW,
           &o_n,
           &o_c,
           &o_d,
           &o_h,
           &o_w);

  int group_offset_in =
      transformed_x.numel() / transformed_x.dims()[0] / groups;
  int group_offset_out =
      transformed_dout.numel() / transformed_dout.dims()[0] / groups;
  int group_offset_filter = filter.numel() / groups;

  ScalingParamType<T> alpha = 1.0f;
  ScalingParamType<T> beta = 0.0f;

  auto workspace_handle = ctx.cudnn_workspace_handle();

  if (ddout) {
    ddx_ = transformed_ddx.data<T>();
#ifdef PADDLE_WITH_HIP
    for (int i = 0; i < groups; i++) {
      workspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenConvolutionBackwardData(
                handle,
                &alpha,
                args1.odesc.desc(),
                ddx_ + i * group_offset_in,
                args1.wdesc.desc(),
                filter_ + i * group_offset_filter,
                args1.cdesc.desc(),
                bwd_result1.algo,
                &beta,
                args1.idesc.desc(),
                transformed_ddout_channel_ + i * group_offset_out,
                workspace_ptr,
                workspace_size));
          },
          workspace_size);
    }
#else   // PADDLE_WITH_HIP
    ConvRunner<T, ConvKind::kBackwardData>::Apply(ctx,
                                                  args1,
                                                  bwd_result1,
                                                  ddx_,
                                                  filter_,
                                                  transformed_ddout_channel_,
                                                  groups,
                                                  group_offset_out,
                                                  group_offset_filter,
                                                  group_offset_in,
                                                  workspace_size,
                                                  &workspace_handle,
                                                  false);
#endif  // PADDLE_WITH_HIP

#ifdef PADDLE_WITH_HIP
    for (int i = 0; i < groups; i++) {
      // MIOPEN ONLY support beta to be 0.0f
      DenseTensor conv_x_ddfilter(dout.type());
      conv_x_ddfilter.Resize(transformed_ddout_channel.dims());
      T* conv_x_ddfilter_data = ctx.template Alloc<T>(&conv_x_ddfilter);
      workspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenConvolutionBackwardData(
                handle,
                &alpha,
                args2.odesc.desc(),
                x_ + i * group_offset_in,
                args2.wdesc.desc(),
                ddfilter_ + i * group_offset_filter,
                args2.cdesc.desc(),
                bwd_result2.algo,
                &beta,
                args2.idesc.desc(),
                conv_x_ddfilter_data + i * group_offset_out,
                workspace_ptr,
                workspace_size));
          },
          workspace_size);
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenOpTensor(
          handle,
          miopenTensorOpAdd,
          &alpha,
          args2.idesc.desc(),
          transformed_ddout_channel_ + i * group_offset_out,
          &alpha,
          args2.idesc.desc(),
          conv_x_ddfilter_data + i * group_offset_out,
          &beta,
          args2.idesc.desc(),
          transformed_ddout_channel_ + i * group_offset_out));
    }
#else   // PADDLE_WITH_HIP
    ConvRunner<T, ConvKind::kBackwardData>::Apply(ctx,
                                                  args2,
                                                  bwd_result2,
                                                  x_,
                                                  ddfilter_,
                                                  transformed_ddout_channel_,
                                                  groups,
                                                  group_offset_out,
                                                  group_offset_filter,
                                                  group_offset_in,
                                                  workspace_size,
                                                  &workspace_handle,
                                                  true);
#endif  // PADDLE_WITH_HIP

    if ((!is_sys_pad) && (!channel_last)) {
      if (strides.size() == 2U) {
        funcs::Slice<Context, T, 4>(
            ctx, &transformed_ddout_channel, ddout, starts, ends, axes);
      } else if (!is_sys_pad && strides.size() == 3U) {
        funcs::Slice<Context, T, 5>(
            ctx, &transformed_ddout_channel, ddout, starts, ends, axes);
      }
    } else if ((!is_sys_pad) && (channel_last)) {
      if (strides.size() == 2U) {
        funcs::Slice<Context, T, 4>(ctx,
                                    &transformed_ddout_channel,
                                    &transformed_ddout_channel,
                                    starts,
                                    ends,
                                    axes);
      } else if (!is_sys_pad && strides.size() == 3U) {
        funcs::Slice<Context, T, 5>(ctx,
                                    &transformed_ddout_channel,
                                    &transformed_ddout_channel,
                                    starts,
                                    ends,
                                    axes);
      }

      TransToChannelLast<Context, T>(ctx, &transformed_ddout_channel, ddout);
    }
  }

  T* transformed_dout_channel_ = transformed_dout.data<T>();
  if (dfilter) {
    ddx_ = transformed_ddx_channel.data<T>();
#ifdef PADDLE_WITH_HIP
    for (int i = 0; i < groups; i++) {
      workspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                dynload::miopenConvolutionBackwardWeights(
                    handle,
                    &alpha,
                    args3.odesc.desc(),
                    ddx_ + i * group_offset_in,
                    args3.idesc.desc(),
                    transformed_dout_channel_ + i * group_offset_out,
                    args3.cdesc.desc(),
                    filter_result.algo,
                    &beta,
                    args3.wdesc.desc(),
                    dfilter_ + i * group_offset_filter,
                    workspace_ptr,
                    workspace_size));
          },
          workspace_size);
    }
#else   // PADDLE_WITH_HIP
    ConvRunner<T, ConvKind::kBackwardFilter>::Apply(ctx,
                                                    args3,
                                                    filter_result,
                                                    ddx_,
                                                    transformed_dout_channel_,
                                                    dfilter_,
                                                    groups,
                                                    group_offset_out,
                                                    group_offset_filter,
                                                    group_offset_in,
                                                    workspace_size,
                                                    &workspace_handle,
                                                    false);
#endif  // PADDLE_WITH_HIP
  }

  if (dx) {
    ddfilter_ = ddfilter.data<T>();
#ifdef PADDLE_WITH_HIP
    for (int i = 0; i < groups; i++) {
      workspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenConvolutionForward(
                handle,
                &alpha,
                args4.idesc.desc(),
                transformed_dout_channel_ + i * group_offset_out,
                args4.wdesc.desc(),
                ddfilter_ + i * group_offset_filter,
                args4.cdesc.desc(),
                fwd_result.algo,
                &beta,
                args4.odesc.desc(),
                transformed_dx_ + i * group_offset_in,
                workspace_ptr,
                workspace_size));
          },
          workspace_size);
    }
#else   // PADDLE_WITH_HIP
    ConvRunner<T, ConvKind::kForward>::Apply(ctx,
                                             args4,
                                             fwd_result,
                                             transformed_dout_channel_,
                                             ddfilter_,
                                             transformed_dx_,
                                             groups,
                                             group_offset_out,
                                             group_offset_filter,
                                             group_offset_in,
                                             workspace_size,
                                             &workspace_handle,
                                             false);
#endif  // PADDLE_WITH_HIP

    if (channel_last) {
      TransToChannelLast<Context, T>(ctx, &transformed_dx_channel, dx);
    }
  }
}

template <typename T, typename Context>
void Conv3dTransposeGradGPUDNNKernel(const Context& ctx,
                                     const DenseTensor& x,
                                     const DenseTensor& filter,
                                     const DenseTensor& dout,
                                     const std::vector<int>& strides,
                                     const std::vector<int>& paddings_,
                                     const std::vector<int>& output_padding,
                                     const std::vector<int>& output_size,
                                     const std::string& padding_algorithm,
                                     int groups,
                                     const std::vector<int>& dilations_,
                                     const std::string& data_format,
                                     DenseTensor* dx,
                                     DenseTensor* dfilter) {
  ConvTransposeGradRawGPUDNNKernel<T, Context>(ctx,
                                               x,
                                               filter,
                                               dout,
                                               strides,
                                               paddings_,
                                               padding_algorithm,
                                               groups,
                                               dilations_,
                                               data_format,
                                               dx,
                                               dfilter);
}

}  // namespace phi

using float16 = phi::dtype::float16;

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(conv2d_transpose_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeGradGPUDNNKernel,
                   float,
                   float16) {}
PD_REGISTER_KERNEL(conv2d_transpose_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeDoubleGradGPUDNNKernel,
                   float,
                   float16) {}
PD_REGISTER_KERNEL(conv3d_transpose_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3dTransposeGradGPUDNNKernel,
                   float,
                   float16) {}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(conv2d_transpose_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeGradGPUDNNKernel,
                   float,
                   double,
                   float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(conv2d_transpose_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeDoubleGradGPUDNNKernel,
                   float,
                   double,
                   float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(conv3d_transpose_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3dTransposeGradGPUDNNKernel,
                   float,
                   double,
                   float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(conv2d_transpose_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeGradGPUDNNKernel,
                   float,
                   double,
                   float16) {}
PD_REGISTER_KERNEL(conv2d_transpose_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeDoubleGradGPUDNNKernel,
                   float,
                   double,
                   float16) {}
PD_REGISTER_KERNEL(conv3d_transpose_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3dTransposeGradGPUDNNKernel,
                   float,
                   double,
                   float16) {}
#endif

#endif
