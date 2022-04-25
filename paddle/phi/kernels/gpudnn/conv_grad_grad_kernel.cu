// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/conv_grad_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/framework/eigen.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/operators/conv_miopen_helper.h"
#else
#include "paddle/fluid/operators/conv_cudnn_helper.h"
#endif

#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/phi/kernels/funcs/padding.h"

#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"

#include "paddle/phi/kernels/impl/conv_cudnn_impl.h"

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void ConvCudnnGradGradKernel(
    const Context& ctx,
    const DenseTensor& input,
    const DenseTensor& filter,
    const DenseTensor& out_grad,
    paddle::optional<const DenseTensor&> input_grad_grad,
    paddle::optional<const DenseTensor&> filter_grad_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings_t,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations_t,
    const std::string& data_format,
    bool use_addto,
    int workspace_size_MB,
    bool exhaustive_search_t,
    DenseTensor* input_grad,
    DenseTensor* filter_grad,
    DenseTensor* out_grad_grad) {
  auto X = &input;
  auto W = &filter;
  auto dO = &out_grad;
  auto ddX = input_grad_grad.get_ptr();
  auto ddW = filter_grad_grad.get_ptr();

  auto ddO = out_grad_grad;
  auto dW = filter_grad;
  auto dX = input_grad;
  if (ddO) {
    ctx.template Alloc<T>(ddO);
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(ctx, ddO, static_cast<T>(0));
  }
  if (dW) {
    ctx.template Alloc<T>(dW);
  }
  if (dX) {
    ctx.template Alloc<T>(dX);
  }

  // const T* x = X->data<T>();
  const T* dy = dO->data<T>();
  const T* w = W->data<T>();

  const T* ddx = nullptr;
  const T* ddw = nullptr;
  T *dw, *dx, *ddy;
  dw = dx = ddy = nullptr;
  T* transformed_dx = nullptr;
  std::vector<int> dilations = dilations_t;

  bool exhaustive_search = FLAGS_cudnn_exhaustive_search || exhaustive_search_t;
  bool deterministic = FLAGS_cudnn_deterministic;
  auto exhaustive_deterministic = exhaustive_search && deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_deterministic,
                    false,
                    phi::errors::InvalidArgument(
                        "Cann't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));

  std::vector<int> paddings = paddings_t;

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  // transform Tensors to channel first-----------
  DenseTensor transformed_X_channel(X->type());
  DenseTensor transformed_dO_channel(dO->type());
  DenseTensor transformed_ddX_channel(X->type());

  DenseTensor transformed_ddO_channel(dO->type());
  DenseTensor transformed_dX_channel(X->type());

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(ctx, X, &transformed_X_channel);
    TransToChannelFirst<Context, T>(ctx, X, &transformed_X_channel);

    ResizeToChannelFirst<Context, T>(ctx, dO, &transformed_dO_channel);
    TransToChannelFirst<Context, T>(ctx, dO, &transformed_dO_channel);

    if (ddX) {
      ResizeToChannelFirst<Context, T>(ctx, ddX, &transformed_ddX_channel);
      TransToChannelFirst<Context, T>(ctx, ddX, &transformed_ddX_channel);
    }

    if (ddO) {
      ResizeToChannelFirst<Context, T>(ctx, ddO, &transformed_ddO_channel);
    }
    if (dX) {
      ResizeToChannelFirst<Context, T>(ctx, dX, &transformed_dX_channel);
      ctx.template Alloc<T>(&transformed_dX_channel);
    }

  } else {
    transformed_X_channel = *X;
    transformed_dO_channel = *dO;
    if (ddX) {
      transformed_ddX_channel = *ddX;
    }
    if (ddO) {
      transformed_ddO_channel.ShareDataWith(*ddO);
    }
    if (dX) {
      transformed_dX_channel.ShareDataWith(*dX);
    }
  }

  auto in_dims = transformed_X_channel.dims();
  auto filter_dims = W->dims();
  DDim in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings, data_dim);
  DenseTensor transformed_X(X->type());
  DenseTensor transformed_ddX(X->type());

  DenseTensor transformed_dX(X->type());

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
    DDim new_input_shape(make_ddim(new_input_shape_vec));
    transformed_X.Resize(new_input_shape);
    transformed_ddX.Resize(new_input_shape);
    transformed_dX.Resize(new_input_shape);

    ctx.template Alloc<T>(&transformed_X);

    if (ddX) {
      ctx.template Alloc<T>(&transformed_ddX);
    }
    if (dX) {
      ctx.template Alloc<T>(&transformed_dX);
    }

    // pad for input
    const int rank = X->dims().size();
    T pad_value(0.0);
    switch (rank) {
      case 4: {
        funcs::PadFunction<Context, T, 4>(
            ctx, input_pad, transformed_X_channel, pad_value, &transformed_X);
        if (ddX) {
          funcs::PadFunction<Context, T, 4>(ctx,
                                            input_pad,
                                            transformed_ddX_channel,
                                            pad_value,
                                            &transformed_ddX);
        }
      } break;
      case 5: {
        funcs::PadFunction<Context, T, 5>(
            ctx, input_pad, transformed_X_channel, pad_value, &transformed_X);
        if (ddX) {
          funcs::PadFunction<Context, T, 5>(ctx,
                                            input_pad,
                                            transformed_ddX_channel,
                                            pad_value,
                                            &transformed_ddX);
        }
      } break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "ConvOp only support tensors with 4 or 5 dimensions."));
    }

  } else {
    transformed_X.ShareDataWith(transformed_X_channel);
    if (ddX) {
      transformed_ddX.ShareDataWith(transformed_ddX_channel);
    }
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
#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
  iwo_group = 1;
  c_group = groups;
  groups = 1;
#endif
  auto dtype = paddle::platform::CudnnDataType<T>::type;

  auto handle = ctx.cudnn_handle();

  paddle::operators::ConvArgs args1{&transformed_ddX,
                                    W,
                                    &transformed_ddO_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype};
  paddle::operators::ConvArgs args2{&transformed_X,
                                    ddW,
                                    &transformed_ddO_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype};
  paddle::operators::ConvArgs args3{&transformed_ddX,
                                    dW,
                                    &transformed_dO_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype};
  paddle::operators::ConvArgs args4{&transformed_dX,
                                    ddW,
                                    &transformed_dO_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype};

#ifdef PADDLE_WITH_HIP
  paddle::operators::SearchResult<miopenConvFwdAlgorithm_t> fwd_result1;
  paddle::operators::SearchResult<miopenConvFwdAlgorithm_t> fwd_result2;
  paddle::operators::SearchResult<miopenConvBwdDataAlgorithm_t> data_result;
  paddle::operators::SearchResult<miopenConvBwdWeightsAlgorithm_t>
      filter_result;
#else
  paddle::operators::SearchResult<cudnnConvolutionFwdAlgo_t> fwd_result1;
  paddle::operators::SearchResult<cudnnConvolutionFwdAlgo_t> fwd_result2;
  paddle::operators::SearchResult<cudnnConvolutionBwdDataAlgo_t> data_result;
  paddle::operators::SearchResult<cudnnConvolutionBwdFilterAlgo_t>
      filter_result;
#endif

  auto layout = paddle::platform::GetCudnnTensorFormat(
      paddle::platform::DataLayout::kNCHW);

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
      args1.cdesc.set(dtype,
                      padding_common,
                      strides,
                      dilations,
                      paddle::platform::AllowTF32Cudnn(),
                      c_group);

#ifdef PADDLE_WITH_HIP
      using search1 =
          paddle::operators::SearchAlgorithm<miopenConvFwdAlgorithm_t>;
      workspace_size = search1::GetWorkspaceSize(args1);
      fwd_result1.algo = search1::Find<T>(
          args1, exhaustive_search, false, workspace_size, ctx);
#else
      using search1 =
          paddle::operators::SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
      fwd_result1 = search1::Find<T>(args1, exhaustive_search, false, ctx);
      workspace_size = search1::GetWorkspaceSize(args1, fwd_result1.algo);
#endif
    }

    if (ddW) {
      ddw = ddW->data<T>();
      args2.handle = handle;
      args2.idesc.set(transformed_X, iwo_group);
      args2.wdesc.set(*ddW, layout, iwo_group);
      args2.odesc.set(transformed_ddO_channel, iwo_group);
      args2.cdesc.set(dtype,
                      padding_common,
                      strides,
                      dilations,
                      paddle::platform::AllowTF32Cudnn(),
                      c_group);

#ifdef PADDLE_WITH_HIP
      using search2 =
          paddle::operators::SearchAlgorithm<miopenConvFwdAlgorithm_t>;
      workspace_size =
          std::max(workspace_size, search2::GetWorkspaceSize(args2));
      fwd_result2.algo = search2::Find<T>(
          args2, exhaustive_search, false, workspace_size, ctx);
#else
      using search2 =
          paddle::operators::SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
      fwd_result2 = search2::Find<T>(args2, exhaustive_search, false, ctx);
      workspace_size = std::max(
          workspace_size, search2::GetWorkspaceSize(args2, fwd_result2.algo));
#endif
    }
  }

  if (dW && ddX) {
    dw = dW->data<T>();
    args3.handle = handle;
    args3.idesc.set(transformed_ddX, iwo_group);
    args3.wdesc.set(*dW, layout, iwo_group);
    args3.odesc.set(transformed_dO_channel, iwo_group);
    args3.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations,
                    paddle::platform::AllowTF32Cudnn(),
                    c_group);

#ifdef PADDLE_WITH_HIP
    using search3 =
        paddle::operators::SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t>;
    workspace_size = std::max(workspace_size, search3::GetWorkspaceSize(args3));
    filter_result.algo = search3::Find<T>(
        args3, exhaustive_search, deterministic, workspace_size, ctx);
#else
    using search3 =
        paddle::operators::SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
    filter_result =
        search3::Find<T>(args3, exhaustive_search, deterministic, ctx);
    workspace_size = std::max(
        workspace_size, search3::GetWorkspaceSize(args3, filter_result.algo));
#endif
  }

  if (ddW && dX) {
    transformed_dx = transformed_dX.data<T>();

    args4.handle = handle;
    args4.idesc.set(transformed_dX, iwo_group);
    args4.wdesc.set(*ddW, layout, iwo_group);
    args4.odesc.set(transformed_dO_channel, iwo_group);
    args4.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations,
                    paddle::platform::AllowTF32Cudnn(),
                    c_group);

#ifdef PADDLE_WITH_HIP
    using search4 =
        paddle::operators::SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
    workspace_size = std::max(workspace_size, search4::GetWorkspaceSize(args4));
    data_result.algo = search4::Find<T>(
        args4, exhaustive_search, deterministic, workspace_size, ctx);
#else
    using search4 =
        paddle::operators::SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
    data_result =
        search4::Find<T>(args4, exhaustive_search, deterministic, ctx);
    workspace_size = std::max(
        workspace_size, search4::GetWorkspaceSize(args4, data_result.algo));
#endif
  }

  int i_n, i_c, i_d, i_h, i_w;
  GetNCDHW(
      transformed_X.dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d, &i_h, &i_w);

  int o_n, o_c, o_d, o_h, o_w;
  GetNCDHW(transformed_dO_channel.dims(),
           DataLayout::kNCHW,
           &o_n,
           &o_c,
           &o_d,
           &o_h,
           &o_w);

  int group_offset_in = i_c / groups * i_h * i_w * i_d;
  int group_offset_out = o_c / groups * o_h * o_w * o_d;
  int group_offset_filter = W->numel() / groups;

  paddle::operators::ScalingParamType<T> alpha = 1.0f;
  paddle::operators::ScalingParamType<T> beta = 0.0f;

  // NOTE(zhiqiu): inplace addto is not supportted in double grad yet.
  // ScalingParamType<T> beta = ctx.Attr<bool>("use_addto") ? 1.0f :
  // 0.0f;
  // VLOG(4) << "Conv_grad_grad: use_addto = " << ctx.Attr<bool>("use_addto");
  auto wkspace_handle = ctx.cudnn_workspace_handle();

  if (ddO) {
    if (ddX) {
      ddx = transformed_ddX.data<T>();
#ifdef PADDLE_WITH_HIP
      wkspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                paddle::platform::dynload::miopenConvolutionForward(
                    handle,
                    &alpha,
                    args1.idesc.desc(),
                    ddx,
                    args1.wdesc.desc(),
                    w,
                    args1.cdesc.desc(),
                    fwd_result1.algo,
                    &beta,
                    args1.odesc.desc(),
                    transformed_ddy_channel,
                    workspace_ptr,
                    workspace_size));
          },
          workspace_size);
#else
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              PADDLE_ENFORCE_GPU_SUCCESS(
                  paddle::platform::dynload::cudnnConvolutionForward(
                      handle,
                      &alpha,
                      args1.idesc.desc(),
                      ddx + i * group_offset_in,
                      args1.wdesc.desc(),
                      w + i * group_offset_filter,
                      args1.cdesc.desc(),
                      fwd_result1.algo,
                      workspace_ptr,
                      workspace_size,
                      &beta,
                      args1.odesc.desc(),
                      transformed_ddy_channel + i * group_offset_out));
            },
            workspace_size);
      }
#endif
    }
    if (ddW) {
#ifdef PADDLE_WITH_HIP
      // MIOPEN ONLY support beta to be 0.0f
      wkspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                paddle::platform::dynload::miopenConvolutionForward(
                    handle,
                    &alpha,
                    args2.idesc.desc(),
                    x,
                    args2.wdesc.desc(),
                    ddw,
                    args2.cdesc.desc(),
                    fwd_result2.algo,
                    &beta,
                    args2.odesc.desc(),
                    transformed_ddy_channel,
                    workspace_ptr,
                    workspace_size));
          },
          workspace_size);
#else
      for (int i = 0; i < groups; i++) {
        wkspace_handle.RunFunc(
            [&](void* workspace_ptr) {
              PADDLE_ENFORCE_GPU_SUCCESS(
                  paddle::platform::dynload::cudnnConvolutionForward(
                      handle,
                      &alpha,
                      args2.idesc.desc(),
                      x + i * group_offset_in,
                      args2.wdesc.desc(),
                      ddw + i * group_offset_filter,
                      args2.cdesc.desc(),
                      fwd_result2.algo,
                      workspace_ptr,
                      workspace_size,
                      &alpha,
                      args2.odesc.desc(),
                      transformed_ddy_channel + i * group_offset_out));
            },
            workspace_size);
      }
#endif
    }
    if (channel_last) {
      TransToChannelLast<Context, T>(ctx, &transformed_ddO_channel, ddO);
    }
  }
  T* transformed_dy_channel = transformed_dO_channel.data<T>();
  if (dW && ddX) {
    ddx = transformed_ddX.data<T>();
#ifdef PADDLE_WITH_HIP
    wkspace_handle.RunFunc(
        [&](void* workspace_ptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              paddle::platform::dynload::miopenConvolutionBackwardWeights(
                  handle,
                  &alpha,
                  args3.odesc.desc(),
                  transformed_dy_channel,
                  args3.idesc.desc(),
                  ddx,
                  args3.cdesc.desc(),
                  filter_result.algo,
                  &beta,
                  args3.wdesc.desc(),
                  dw,
                  workspace_ptr,
                  workspace_size));
        },
        workspace_size);
#else
    for (int i = 0; i < groups; i++) {
      wkspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                paddle::platform::dynload::cudnnConvolutionBackwardFilter(
                    handle,
                    &alpha,
                    args3.idesc.desc(),
                    ddx + i * group_offset_in,
                    args3.odesc.desc(),
                    transformed_dy_channel + i * group_offset_out,
                    args3.cdesc.desc(),
                    filter_result.algo,
                    workspace_ptr,
                    workspace_size,
                    &beta,
                    args3.wdesc.desc(),
                    dw + i * group_offset_filter));
          },
          workspace_size);
    }
#endif
  }

  if (dX && ddW) {
    ddw = ddW->data<T>();
#ifdef PADDLE_WITH_HIP
    wkspace_handle.RunFunc(
        [&](void* workspace_ptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              paddle::platform::dynload::miopenConvolutionBackwardData(
                  handle,
                  &alpha,
                  args4.odesc.desc(),
                  transformed_dy_channel,
                  args4.wdesc.desc(),
                  ddw,
                  args4.cdesc.desc(),
                  data_result.algo,
                  &beta,
                  args4.idesc.desc(),
                  transformed_dx,
                  workspace_ptr,
                  workspace_size));
        },
        workspace_size);
#else
    for (int i = 0; i < groups; i++) {
      wkspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                paddle::platform::dynload::cudnnConvolutionBackwardData(
                    handle,
                    &alpha,
                    args4.wdesc.desc(),
                    ddw + i * group_offset_filter,
                    args4.odesc.desc(),
                    transformed_dy_channel + i * group_offset_out,
                    args4.cdesc.desc(),
                    data_result.algo,
                    workspace_ptr,
                    workspace_size,
                    &beta,
                    args4.idesc.desc(),
                    transformed_dx + i * group_offset_in));
          },
          workspace_size);
    }
#endif

    if (!is_sys_pad) {
      // reverse padded input
      std::vector<int> starts(X->dims().size(), 0);
      std::vector<int> axes(X->dims().size(), 0);

      for (size_t i = 0; i < X->dims().size(); ++i) {
        starts[i] = input_pad[2 * i];
        axes[i] = i;
      }
      if (X->dims().size() == 4) {
        paddle::operators::RemovePaddingSlice<Context, T, 4>(
            ctx, &transformed_dX, &transformed_dX_channel, starts, axes);
      } else {
        paddle::operators::RemovePaddingSlice<Context, T, 5>(
            ctx, &transformed_dX, &transformed_dX_channel, starts, axes);
      }
    }
    if (channel_last) {
      TransToChannelLast<Context, T>(ctx, &transformed_dX_channel, dX);
    }
  }
}

template <typename T, typename Context>
void DepthwiseConvCudnnGradGradKernel(
    const Context& ctx,
    paddle::optional<const DenseTensor&> input_grad_grad,
    paddle::optional<const DenseTensor&> filter_grad_grad,
    const DenseTensor& out_grad,
    const DenseTensor& input,
    const DenseTensor& filter,
    const std::vector<int>& strides,
    const std::vector<int>& paddings_t,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations_t,
    const std::string& data_format,
    bool use_addto,
    int workspace_size_MB,
    bool exhaustive_search_t,
    bool fuse_relu,
    DenseTensor* out_grad_grad,
    DenseTensor* input_grad,
    DenseTensor* filter_grad) {
  ConvCudnnGradGradKernel<T>(ctx,
                             input,
                             filter,
                             out_grad,
                             input_grad_grad,
                             filter_grad_grad,
                             strides,
                             paddings_t,
                             padding_algorithm,
                             groups,
                             dilations_t,
                             data_format,
                             use_addto,
                             workspace_size_MB,
                             exhaustive_search_t,
                             input_grad,
                             filter_grad,
                             out_grad_grad);
}

template <typename T, typename Context>
void Conv3DCudnnGradGradKernel(
    const Context& ctx,
    paddle::optional<const DenseTensor&> input_grad_grad,
    paddle::optional<const DenseTensor&> filter_grad_grad,
    const DenseTensor& out_grad,
    const DenseTensor& input,
    const DenseTensor& filter,
    const std::vector<int>& strides,
    const std::vector<int>& paddings_t,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations_t,
    const std::string& data_format,
    bool use_addto,
    int workspace_size_MB,
    bool exhaustive_search_t,
    DenseTensor* out_grad_grad,
    DenseTensor* input_grad,
    DenseTensor* filter_grad) {
  ConvCudnnGradGradKernel<T>(ctx,
                             input,
                             filter,
                             out_grad,
                             input_grad_grad,
                             filter_grad_grad,
                             strides,
                             paddings_t,
                             padding_algorithm,
                             groups,
                             dilations_t,
                             data_format,
                             use_addto,
                             workspace_size_MB,
                             exhaustive_search_t,
                             input_grad,
                             filter_grad,
                             out_grad_grad);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(conv2d_grad_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d_grad_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnGradGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(depthwise_conv2d_grad_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvCudnnGradGradKernel,
                   float,
                   phi::dtype::float16) {}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(conv2d_grad_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(conv3d_grad_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnGradGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(depthwise_conv2d_grad_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvCudnnGradGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

#else

PD_REGISTER_KERNEL(conv2d_grad_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d_grad_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnGradGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(depthwise_conv2d_grad_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvCudnnGradGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

#endif

#endif
