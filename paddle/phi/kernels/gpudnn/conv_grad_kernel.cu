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

#include "paddle/phi/kernels/conv_grad_kernel.h"

#include "paddle/fluid/framework/eigen.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/operators/conv_miopen_helper.h"
#else
#include "paddle/fluid/operators/conv_cudnn_helper.h"
#endif

#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/padding.h"
#include "paddle/phi/kernels/impl/conv_cudnn_impl.h"

namespace phi {

template <typename T, typename Context>
void ConvCudnnGradKernel(const Context& ctx,
                         const DenseTensor& input,
                         const DenseTensor& filter,
                         const DenseTensor& output_grad,
                         const std::vector<int>& strides_t,
                         const std::vector<int>& paddings_t,
                         const std::string& padding_algorithm,
                         const std::vector<int>& dilations_t,
                         int groups,
                         const std::string& data_format,
                         DenseTensor* input_grad,
                         DenseTensor* filter_grad) {
  if (input_grad) {
    ctx.template Alloc<T>(input_grad);
  }
  if (filter_grad) {
    ctx.template Alloc<T>(filter_grad);
  }

  bool has_use_addto = ctx.HasDnnAttr("use_addto");
  VLOG(4) << "GPUContext contains `use_addto`: " << has_use_addto;
  bool use_addto = has_use_addto
                       ? PADDLE_GET_CONST(bool, ctx.GetDnnAttr("use_addto"))
                       : false;

  std::vector<int> dilations = dilations_t;
  std::vector<int> strides = strides_t;
  std::vector<int> paddings = paddings_t;

  bool has_exhaustive_search = ctx.HasDnnAttr("exhaustive_search");
  VLOG(4) << "GPUContext contains `exhaustive_search`: "
          << has_exhaustive_search;
  bool exhaustive_search_attr =
      has_exhaustive_search
          ? PADDLE_GET_CONST(bool, ctx.GetDnnAttr("exhaustive_search"))
          : false;
  bool exhaustive_search =
      FLAGS_cudnn_exhaustive_search || exhaustive_search_attr;
  bool deterministic = FLAGS_cudnn_deterministic;
  auto exhaustive_deterministic = exhaustive_search && deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_deterministic,
                    false,
                    phi::errors::InvalidArgument(
                        "Cann't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  auto dtype = paddle::platform::CudnnDataType<T>::type;

#ifdef PADDLE_WITH_HIP
  // HIP MIOPEN ONLY SUPPORT NCHW format
  auto compute_format = paddle::platform::DataLayout::kNCHW;
#else
  const bool compute_in_nhwc = dtype == CUDNN_DATA_HALF && IsVoltaOrLater(ctx);
  auto compute_format = compute_in_nhwc && channel_last
                            ? paddle::platform::DataLayout::kNHWC
                            : paddle::platform::DataLayout::kNCHW;
#endif
  VLOG(3) << "Compute ConvGradOp with cuDNN:"
          << " data_format=" << data_format << " compute_format="
          << (compute_format == paddle::platform::DataLayout::kNHWC ? "NHWC"
                                                                    : "NCHW");

  // transform Tensor
  DenseTensor transformed_input_channel(input.type());
  DenseTensor transformed_output_grad_channel(output_grad.type());
  DenseTensor transformed_input_grad_channel(input.type());
  DenseTensor transformed_filter_channel(filter.type());
  DenseTensor transformed_filter_grad_channel(filter.type());

  if (channel_last && compute_format == paddle::platform::DataLayout::kNCHW) {
    VLOG(3) << "Transform input, output_grad, input_grad and tensor from "
               "NHWC to NCHW.";
    ResizeToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);
    TransToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);

    ResizeToChannelFirst<Context, T>(
        ctx, &output_grad, &transformed_output_grad_channel);
    TransToChannelFirst<Context, T>(
        ctx, &output_grad, &transformed_output_grad_channel);

    if (input_grad) {
      ResizeToChannelFirst<Context, T>(
          ctx, input_grad, &transformed_input_grad_channel);
      // NOTE(zhiqiu): If inplace_addto strategy is enabled, we need to copy
      // the data of input_grad to transformed_input_grad_channel.
      if (use_addto) {
        TransToChannelFirst<Context, T>(
            ctx, input_grad, &transformed_input_grad_channel);
      }
    }
  } else {
    transformed_input_channel.ShareDataWith(input);
    transformed_output_grad_channel.ShareDataWith(output_grad);
    if (input_grad) {
      transformed_input_grad_channel.ShareDataWith(*input_grad);
    }
  }

  if (compute_format == paddle::platform::DataLayout::kNHWC) {
    VLOG(3) << "Transform filter and filter_grad tensor from NCHW to NHWC.";
    ResizeToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);
    TransToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);

    if (filter_grad) {
      ResizeToChannelLast<Context, T>(
          ctx, filter_grad, &transformed_filter_grad_channel);
    }
  } else {
    transformed_filter_channel.ShareDataWith(filter);
    if (filter_grad) {
      transformed_filter_grad_channel.ShareDataWith(*filter_grad);
    }
  }

  //  update paddings
  auto in_dims = transformed_input_channel.dims();
  auto filter_dims = transformed_filter_channel.dims();
  DDim in_data_dims;
  DDim filter_data_dims;
  if (compute_format == paddle::platform::DataLayout::kNCHW) {
    in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
    filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  } else {
    in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
    filter_data_dims = slice_ddim(filter_dims, 1, filter_dims.size() - 1);
  }
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  // cuDNN only supports padding the same amount on every dimension.
  // So we create a new padded input tensor.
  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings, data_dim);
  Tensor transformed_input(input.type());
  Tensor transformed_input_grad(input.type());
  std::vector<int> padding_common(data_dim, 0);
  std::vector<int> input_pad(transformed_input_channel.dims().size() * 2, 0);

  if (!is_sys_pad) {
    // get pad
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_input_shape_vec(data_dim + 2);
    new_input_shape_vec[0] = transformed_input_channel.dims()[0];
    if (compute_format == paddle::platform::DataLayout::kNCHW) {
      new_input_shape_vec[1] = transformed_input_channel.dims()[1];
    } else {
      new_input_shape_vec[data_dim + 1] =
          transformed_input_channel.dims()[data_dim + 1];
    }

    for (size_t i = 0; i < data_dim; ++i) {
      padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
      padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
      if (compute_format == paddle::platform::DataLayout::kNCHW) {
        new_input_shape_vec[i + 2] =
            transformed_input_channel.dims()[i + 2] + padding_diff[i];
      } else {
        new_input_shape_vec[i + 1] =
            transformed_input_channel.dims()[i + 1] + padding_diff[i];
      }
      if (compute_format == paddle::platform::DataLayout::kNCHW) {
        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      } else {
        input_pad[2 * i + 2] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 2 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
    }
    DDim new_input_shape(make_ddim(new_input_shape_vec));
    transformed_input.Resize(new_input_shape);
    ctx.template Alloc<T>(&transformed_input);

    transformed_input_grad.Resize(new_input_shape);

    if (input_grad) {
      ctx.template Alloc<T>(&transformed_input_grad);
    }
    // pad for input
    const int rank = transformed_input_channel.dims().size();
    T pad_value(0.0);
    switch (rank) {
      case 4: {
        funcs::PadFunction<Context, T, 4>(ctx,
                                          input_pad,
                                          transformed_input_channel,
                                          pad_value,
                                          &transformed_input);
      } break;
      case 5: {
        funcs::PadFunction<Context, T, 5>(ctx,
                                          input_pad,
                                          transformed_input_channel,
                                          pad_value,
                                          &transformed_input);
      } break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "ConvOp only support tensors with 4 or 5 dimensions."));
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
  const T* filter_data = transformed_filter_channel.data<T>();
  T* filter_grad_data = nullptr;
  T* input_grad_data = nullptr;
  T* transformed_input_grad_data = nullptr;

  paddle::platform::DataLayout layout =
      compute_format == paddle::platform::DataLayout::kNHWC
          ? paddle::platform::DataLayout::kNHWC
          : paddle::platform::DataLayout::kNCHW;

  paddle::operators::ConvArgs args1{&transformed_input_grad,
                                    &transformed_filter_channel,
                                    &transformed_output_grad_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype,
                                    groups,
                                    layout};
  paddle::operators::ConvArgs args2{&transformed_input,
                                    &transformed_filter_grad_channel,
                                    &transformed_output_grad_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype,
                                    groups,
                                    layout};

  auto handle = ctx.cudnn_handle();
  // TODO(phlrain): replace paddle::platform::DataLaytout to phi::DataLayout

  if (transformed_input.dims().size() == 5) {
    layout = compute_format == paddle::platform::DataLayout::kNHWC
                 ? paddle::platform::DataLayout::kNDHWC
                 : paddle::platform::DataLayout::kNCDHW;
  }
  auto layout_tensor = paddle::platform::GetCudnnTensorFormat(layout);
  auto workspace_handle = ctx.cudnn_workspace_handle();

  int i_n, i_c, i_d, i_h, i_w;
  int o_n, o_c, o_d, o_h, o_w;
  if (compute_format == paddle::platform::DataLayout::kNHWC) {
    paddle::operators::GetNCDHW(transformed_input.dims(),
                                paddle::platform::DataLayout::kNHWC,
                                &i_n,
                                &i_c,
                                &i_d,
                                &i_h,
                                &i_w);
    paddle::operators::GetNCDHW(transformed_output_grad_channel.dims(),
                                paddle::platform::DataLayout::kNHWC,
                                &o_n,
                                &o_c,
                                &o_d,
                                &o_h,
                                &o_w);
  } else {
    paddle::operators::GetNCDHW(transformed_input.dims(),
                                paddle::platform::DataLayout::kNCHW,
                                &i_n,
                                &i_c,
                                &i_d,
                                &i_h,
                                &i_w);
    paddle::operators::GetNCDHW(transformed_output_grad_channel.dims(),
                                paddle::platform::DataLayout::kNCHW,
                                &o_n,
                                &o_c,
                                &o_d,
                                &o_h,
                                &o_w);
  }

  int group_offset_in = i_c / groups * i_h * i_w * i_d;
  int group_offset_out = o_c / groups * o_h * o_w * o_d;
  int group_offset_filter = transformed_filter_channel.numel() / groups;

// ------------------- cudnn backward algorithm ---------------------
#ifdef PADDLE_WITH_HIP
  paddle::operators::SearchResult<miopenConvBwdDataAlgorithm_t> bwd_result;
  paddle::operators::SearchResult<miopenConvBwdWeightsAlgorithm_t>
      filter_result;
#else
  paddle::operators::SearchResult<cudnnConvolutionBwdDataAlgo_t> bwd_result;
  paddle::operators::SearchResult<cudnnConvolutionBwdFilterAlgo_t>
      filter_result;
#endif
  // input data workspace_size
  size_t workspace_size_d = 0;
  // weight workspace_size
  size_t workspace_size_w = 0;
  int iwo_groups = groups;
  int c_groups = 1;

#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
  iwo_groups = 1;
  c_groups = groups;
  groups = 1;
#endif

  if (input_grad) {
    // ------------------- cudnn descriptors ---------------------
    input_grad_data = input_grad->data<T>();
    transformed_input_grad_data = transformed_input_grad.data<T>();

    args1.handle = handle;
    args1.idesc.set(transformed_input_grad, layout_tensor);
    args1.wdesc.set(transformed_filter_channel, layout_tensor, iwo_groups);
    args1.odesc.set(transformed_output_grad_channel, layout_tensor);
    args1.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations,
                    paddle::platform::AllowTF32Cudnn(),
                    c_groups);

#ifdef PADDLE_WITH_HIP
    using search1 =
        paddle::operators::SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
    workspace_size_d =
        std::max(workspace_size_d, search1::GetWorkspaceSize(args1));
    bwd_result.algo = search1::Find<T>(
        args1, exhaustive_search, deterministic, workspace_size_d, ctx);
#else
    using search1 =
        paddle::operators::SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
    bwd_result = search1::Find<T>(args1, exhaustive_search, deterministic, ctx);
    workspace_size_d = std::max(workspace_size_d, bwd_result.workspace_size);
#endif
  }

  if (filter_grad) {
    // ------------------- cudnn descriptors ---------------------
    filter_grad_data = transformed_filter_grad_channel.data<T>();
    args2.handle = handle;
    args2.idesc.set(transformed_input, layout_tensor);
    args2.wdesc.set(transformed_filter_grad_channel, layout_tensor, iwo_groups);
    args2.odesc.set(transformed_output_grad_channel, layout_tensor);
    args2.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations,
                    paddle::platform::AllowTF32Cudnn(),
                    c_groups);
#ifdef PADDLE_WITH_HIP
    using search2 =
        paddle::operators::SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t>;
    workspace_size_w =
        std::max(workspace_size_w, search2::GetWorkspaceSize(args2));
    filter_result.algo = search2::Find<T>(
        args2, exhaustive_search, deterministic, workspace_size_w, ctx);
#else
    using search2 =
        paddle::operators::SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
    filter_result =
        search2::Find<T>(args2, exhaustive_search, deterministic, ctx);
    VLOG(3) << "filter algo: " << filter_result.algo << ", time "
            << filter_result.time;
    workspace_size_w = std::max(workspace_size_w, filter_result.workspace_size);
#endif
  }

  // ------------------- cudnn conv backward data ---------------------
  paddle::operators::ScalingParamType<T> alpha = 1.0f;
#ifdef PADDLE_WITH_HIP
  // MIOPEN ONLY support beta to be 0.0f
  paddle::operators::ScalingParamType<T> beta = 0.0f;
#else
  paddle::operators::ScalingParamType<T> beta = use_addto ? 1.0f : 0.0f;

#endif
  VLOG(4) << "Conv_grad: use_addto = " << use_addto;

  if (input_grad) {
// When beta is 0, it is unnecessary to reset input_grad.
// When beta is 1, the output cannot be reset since addt strategy used.
#ifdef PADDLE_WITH_HIP
    if (use_addto) {
      DenseTensor temp_tensor(transformed_input_grad.type());
      temp_tensor.Resize(transformed_input_grad.dims());
      T* temp_tensor_data = ctx.template Alloc<T>(&temp_tensor);
      workspace_handle.RunFunc(
          [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                paddle::platform::dynload::miopenConvolutionBackwardData(
                    handle,
                    &alpha,
                    args1.odesc.desc(),
                    output_grad_data,
                    args1.wdesc.desc(),
                    filter_data,
                    args1.cdesc.desc(),
                    bwd_result.algo,
                    &beta,
                    args1.idesc.desc(),
                    temp_tensor_data,
                    cudnn_workspace_ptr,
                    workspace_size_d));
          },
          workspace_size_d);
      PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::miopenOpTensor(
          handle,
          miopenTensorOpAdd,
          &alpha,
          args1.idesc.desc(),
          transformed_input_grad_data,
          &alpha,
          args1.idesc.desc(),
          temp_tensor_data,
          &beta,
          args1.idesc.desc(),
          transformed_input_grad_data));
    } else {
      workspace_handle.RunFunc(
          [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                paddle::platform::dynload::miopenConvolutionBackwardData(
                    handle,
                    &alpha,
                    args1.odesc.desc(),
                    output_grad_data,
                    args1.wdesc.desc(),
                    filter_data,
                    args1.cdesc.desc(),
                    bwd_result.algo,
                    &beta,
                    args1.idesc.desc(),
                    transformed_input_grad_data,
                    cudnn_workspace_ptr,
                    workspace_size_d));
          },
          workspace_size_d);
    }

#else
    for (int i = 0; i < groups; i++) {
      workspace_handle.RunFunc(
          [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                paddle::platform::dynload::cudnnConvolutionBackwardData(
                    handle,
                    &alpha,
                    args1.wdesc.desc(),
                    filter_data + i * group_offset_filter,
                    args1.odesc.desc(),
                    output_grad_data + i * group_offset_out,
                    args1.cdesc.desc(),
                    bwd_result.algo,
                    cudnn_workspace_ptr,
                    workspace_size_d,
                    &beta,
                    args1.idesc.desc(),
                    transformed_input_grad_data + i * group_offset_in));
          },
          workspace_size_d);
    }
#endif
    if (!is_sys_pad) {
      std::vector<int> starts(transformed_input_channel.dims().size(), 0);
      std::vector<int> axes(transformed_input_channel.dims().size(), 0);

      for (size_t i = 0; i < transformed_input_channel.dims().size(); ++i) {
        starts[i] = input_pad[2 * i];
        axes[i] = i;
      }

      ctx.template Alloc<T>(&transformed_input_grad_channel);
      if (transformed_input_channel.dims().size() == 4) {
        paddle::operators::RemovePaddingSlice<Context, T, 4>(
            ctx,
            &transformed_input_grad,
            &transformed_input_grad_channel,
            starts,
            axes);
      } else {
        paddle::operators::RemovePaddingSlice<Context, T, 5>(
            ctx,
            &transformed_input_grad,
            &transformed_input_grad_channel,
            starts,
            axes);
      }
    }

    if (channel_last && compute_format == paddle::platform::DataLayout::kNCHW) {
      TransToChannelLast<Context, T>(
          ctx, &transformed_input_grad_channel, input_grad);
    }
  }

  // filter_grad do not use inplace addto.
  paddle::operators::ScalingParamType<T> beta_filter = 0.0f;
  // ------------------- cudnn conv backward filter ---------------------
  if (filter_grad) {
// Because beta is zero, it is unnecessary to reset filter_grad.
#ifdef PADDLE_WITH_HIP
    workspace_handle.RunFunc(
        [&](void* cudnn_workspace_ptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              paddle::platform::dynload::miopenConvolutionBackwardWeights(
                  handle,
                  &alpha,
                  args2.odesc.desc(),
                  output_grad_data,
                  args2.idesc.desc(),
                  input_data,
                  args2.cdesc.desc(),
                  filter_result.algo,
                  &beta,
                  args2.wdesc.desc(),
                  filter_grad_data,
                  cudnn_workspace_ptr,
                  workspace_size_w));
        },
        workspace_size_w);
#else
    for (int i = 0; i < groups; i++) {
      workspace_handle.RunFunc(
          [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                paddle::platform::dynload::cudnnConvolutionBackwardFilter(
                    handle,
                    &alpha,
                    args2.idesc.desc(),
                    input_data + i * group_offset_in,
                    args2.odesc.desc(),
                    output_grad_data + i * group_offset_out,
                    args2.cdesc.desc(),
                    filter_result.algo,
                    cudnn_workspace_ptr,
                    workspace_size_w,
                    &beta_filter,
                    args2.wdesc.desc(),
                    filter_grad_data + i * group_offset_filter));
          },
          workspace_size_w);
    }
#endif

    if (compute_format == paddle::platform::DataLayout::kNHWC) {
      TransToChannelFirst<Context, T>(
          ctx, &transformed_filter_grad_channel, filter_grad);
    }
  }
}

template <typename T, typename Context>
void Conv3DCudnnGradKernel(const Context& dev_ctx,
                           const DenseTensor& input,
                           const DenseTensor& filter,
                           const DenseTensor& out_grad,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::string& paddding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           bool use_addto,
                           int workspace_size_MB,
                           bool exhaustive_search,
                           DenseTensor* input_grad,
                           DenseTensor* filter_grad) {
  ConvCudnnGradKernel<T>(dev_ctx,
                         input,
                         filter,
                         out_grad,
                         strides,
                         paddings,
                         paddding_algorithm,
                         dilations,
                         groups,
                         data_format,
                         input_grad,
                         filter_grad);
}

template <typename T, typename Context>
void DepthwiseConvCudnnGradKernel(const Context& dev_ctx,
                                  const DenseTensor& input,
                                  const DenseTensor& filter,
                                  const DenseTensor& out_grad,
                                  const std::vector<int>& strides,
                                  const std::vector<int>& paddings,
                                  const std::string& paddding_algorithm,
                                  int groups,
                                  const std::vector<int>& dilations,
                                  const std::string& data_format,
                                  bool use_addto,
                                  int workspace_size_MB,
                                  bool exhaustive_search,
                                  bool fuse_relu,
                                  DenseTensor* input_grad,
                                  DenseTensor* filter_grad) {
  ConvCudnnGradKernel<T>(dev_ctx,
                         input,
                         filter,
                         out_grad,
                         strides,
                         paddings,
                         paddding_algorithm,
                         dilations,
                         groups,
                         data_format,
                         input_grad,
                         filter_grad);
}

template <typename T, typename Context>
void ConvCudnnGradGradKernel(
    const Context& ctx,
    const DenseTensor& input,
    const DenseTensor& filter,
    const DenseTensor& out_grad,
    const paddle::optional<DenseTensor>& input_grad_grad,
    const paddle::optional<DenseTensor>& filter_grad_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings_t,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations_t,
    int groups,
    const std::string& data_format,
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

  bool has_exhaustive_search = ctx.HasDnnAttr("exhaustive_search");
  VLOG(4) << "GPUContext contains `exhaustive_search`: "
          << has_exhaustive_search;
  bool exhaustive_search_attr =
      has_exhaustive_search
          ? PADDLE_GET_CONST(bool, ctx.GetDnnAttr("exhaustive_search"))
          : false;
  bool exhaustive_search =
      FLAGS_cudnn_exhaustive_search || exhaustive_search_attr;
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
  auto layout = paddle::platform::GetCudnnTensorFormat(
      paddle::platform::DataLayout::kNCHW);

  paddle::operators::ConvArgs args1{&transformed_ddX,
                                    W,
                                    &transformed_ddO_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype,
                                    groups,
                                    paddle::platform::DataLayout::kNCHW};
  paddle::operators::ConvArgs args2{&transformed_X,
                                    ddW,
                                    &transformed_ddO_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype,
                                    groups,
                                    paddle::platform::DataLayout::kNCHW};
  paddle::operators::ConvArgs args3{&transformed_ddX,
                                    dW,
                                    &transformed_dO_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype,
                                    groups,
                                    paddle::platform::DataLayout::kNCHW};
  paddle::operators::ConvArgs args4{&transformed_dX,
                                    ddW,
                                    &transformed_dO_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype,
                                    groups,
                                    paddle::platform::DataLayout::kNCHW};

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
void DepthwiseConvDoubleGradGPUDNNKernel(
    const Context& ctx,
    const DenseTensor& input,
    const DenseTensor& filter,
    const DenseTensor& out_grad,
    const paddle::optional<DenseTensor>& input_grad_grad,
    const paddle::optional<DenseTensor>& filter_grad_grad,
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
    DenseTensor* input_grad,
    DenseTensor* filter_grad,
    DenseTensor* out_grad_grad) {
  ConvCudnnGradGradKernel<T>(ctx,
                             input,
                             filter,
                             out_grad,
                             input_grad_grad,
                             filter_grad_grad,
                             strides,
                             paddings_t,
                             padding_algorithm,
                             dilations_t,
                             groups,
                             data_format,
                             input_grad,
                             filter_grad,
                             out_grad_grad);
}

template <typename T, typename Context>
void Conv3DCudnnGradGradKernel(
    const Context& ctx,
    const DenseTensor& input,
    const DenseTensor& filter,
    const DenseTensor& out_grad,
    const paddle::optional<DenseTensor>& input_grad_grad,
    const paddle::optional<DenseTensor>& filter_grad_grad,
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
  ConvCudnnGradGradKernel<T>(ctx,
                             input,
                             filter,
                             out_grad,
                             input_grad_grad,
                             filter_grad_grad,
                             strides,
                             paddings_t,
                             padding_algorithm,
                             dilations_t,
                             groups,
                             data_format,
                             input_grad,
                             filter_grad,
                             out_grad_grad);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(conv2d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(depthwise_conv2d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::DepthwiseConvCudnnGradKernel,
                   float,
                   phi::dtype::float16) {}
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
                   phi::DepthwiseConvDoubleGradGPUDNNKernel,
                   float,
                   phi::dtype::float16) {}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(conv2d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(conv3d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
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
                   phi::DepthwiseConvDoubleGradGPUDNNKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(conv2d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

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
                   phi::DepthwiseConvDoubleGradGPUDNNKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif

#endif
