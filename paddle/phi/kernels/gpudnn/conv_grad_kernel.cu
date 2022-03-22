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

#include "paddle/phi/core/dense_tensor.h"

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

namespace phi {

template <typename T, typename Context>
void ConvCudnnGradKernel(const Context& ctx,
                         const DenseTensor& output_grad,
                         const DenseTensor& input,
                         const DenseTensor& filter,
                         const std::vector<int>& strides_t,
                         const std::vector<int>& paddings_t,
                         const std::string& padding_algorithm,
                         int groups,
                         const std::vector<int>& dilations_t,
                         const std::string& data_format,
                         bool use_addto,
                         int workspace_size_MB,
                         bool exhaustive_search_t,
                         DenseTensor* input_grad,
                         DenseTensor* filter_grad) {
  if (input_grad) {
    ctx.template Alloc<T>(input_grad);
  }
  if (filter_grad) {
    ctx.template Alloc<T>(filter_grad);
  }

  std::vector<int> dilations = dilations_t;
  std::vector<int> strides = strides_t;
  std::vector<int> paddings = paddings_t;

  bool exhaustive_search = FLAGS_cudnn_exhaustive_search || exhaustive_search_t;
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

  paddle::operators::ConvArgs args1{&transformed_input_grad,
                                    &transformed_filter_channel,
                                    &transformed_output_grad_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype};
  paddle::operators::ConvArgs args2{&transformed_input,
                                    &transformed_filter_grad_channel,
                                    &transformed_output_grad_channel,
                                    strides,
                                    padding_common,
                                    dilations,
                                    dtype};

  auto handle = ctx.cudnn_handle();
  // TODO(phlrain): replace paddle::platform::DataLaytout to phi::DataLayout
  paddle::platform::DataLayout layout =
      compute_format == paddle::platform::DataLayout::kNHWC
          ? paddle::platform::DataLayout::kNHWC
          : paddle::platform::DataLayout::kNCHW;
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
  miopenConvBwdDataAlgorithm_t data_algo =
      static_cast<miopenConvBwdDataAlgorithm_t>(0);
  miopenConvBwdWeightsAlgorithm_t filter_algo =
      static_cast<miopenConvBwdWeightsAlgorithm_t>(0);
#else
  cudnnConvolutionBwdDataAlgo_t data_algo =
      static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
  cudnnConvolutionBwdFilterAlgo_t filter_algo =
      static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);
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
    data_algo = search1::Find<T>(
        args1, exhaustive_search, deterministic, workspace_size_d, ctx);
#else
    using search1 =
        paddle::operators::SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
    data_algo = search1::Find<T>(args1, exhaustive_search, deterministic, ctx);
    workspace_size_d =
        std::max(workspace_size_d, search1::GetWorkspaceSize(args1, data_algo));
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
    filter_algo = search2::Find<T>(
        args2, exhaustive_search, deterministic, workspace_size_w, ctx);
#else
    using search2 =
        paddle::operators::SearchAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>;
    filter_algo =
        search2::Find<T>(args2, exhaustive_search, deterministic, ctx);
    workspace_size_w = std::max(workspace_size_w,
                                search2::GetWorkspaceSize(args2, filter_algo));
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
                    data_algo,
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
                    data_algo,
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
                    data_algo,
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
                  filter_algo,
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
                    filter_algo,
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
                           const DenseTensor& out_grad,
                           const DenseTensor& input,
                           const DenseTensor& filter,
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
                         out_grad,
                         input,
                         filter,
                         strides,
                         paddings,
                         paddding_algorithm,
                         groups,
                         dilations,
                         data_format,
                         use_addto,
                         workspace_size_MB,
                         exhaustive_search,
                         input_grad,
                         filter_grad);
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

#endif

#endif
