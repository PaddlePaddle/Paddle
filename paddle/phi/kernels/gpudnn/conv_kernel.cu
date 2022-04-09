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
#include "paddle/phi/kernels/conv_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

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
void ConvCudnnKernel(const Context& ctx,
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
                     DenseTensor* output) {
  ctx.template Alloc<T>(output);
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;

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
  // Tensor Core introduced from Volta GPUs supports more faster conv op
  // with FP16 in NHWC data format.
  const bool compute_in_nhwc = dtype == CUDNN_DATA_HALF && IsVoltaOrLater(ctx);
  // We will only do data format conversion from NHWC to NCHW.
  // cudnn will convert NCHW to NHWC automatically on Tensor Core.
  auto compute_format = compute_in_nhwc && channel_last
                            ? paddle::platform::DataLayout::kNHWC
                            : paddle::platform::DataLayout::kNCHW;
#endif
  VLOG(3) << "Compute ConvOp with cuDNN:"
          << " data_format=" << data_format << " compute_format="
          << (compute_format == paddle::platform::DataLayout::kNHWC ? "NHWC"
                                                                    : "NCHW");

  // ------------ transformed tensor -----------
  DenseTensor transformed_input_channel(input.type());
  DenseTensor transformed_output(output->type());
  DenseTensor transformed_filter_channel(filter.type());
  T* output_data = nullptr;
  if (channel_last && compute_format == paddle::platform::DataLayout::kNCHW) {
    VLOG(3) << "Transform input tensor from NHWC to NCHW.";
    ResizeToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);
    TransToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);

    ResizeToChannelFirst<Context, T>(ctx, output, &transformed_output);

  } else {
    transformed_input_channel.ShareDataWith(input);
    transformed_output.ShareDataWith(*output);
  }
  if (compute_format == paddle::platform::DataLayout::kNHWC) {
    VLOG(3) << "Transform filter tensor from NCHW to NHWC.";
    ResizeToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);
    TransToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);
  } else {
    transformed_filter_channel.ShareDataWith(filter);
  }
  output_data = transformed_output.data<T>();

  // update padding and dilation
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

  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings, data_dim);

  DenseTensor transformed_input;
  std::vector<int> padding_common(data_dim, 0);
  if (!is_sys_pad) {
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_input_shape_vec(data_dim + 2);
    new_input_shape_vec[0] = transformed_input_channel.dims()[0];

    if (compute_format == paddle::platform::DataLayout::kNCHW) {
      new_input_shape_vec[1] = transformed_input_channel.dims()[1];
    } else {
      new_input_shape_vec[data_dim + 1] =
          transformed_input_channel.dims()[data_dim + 1];
    }

    std::vector<int> input_pad(transformed_input_channel.dims().size() * 2, 0);
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

  const T* filter_data = transformed_filter_channel.data<T>();

  // ------------------- cudnn descriptors ---------------------
  paddle::operators::ConvArgs args{&transformed_input,
                                   &transformed_filter_channel,
                                   &transformed_output,
                                   strides,
                                   padding_common,
                                   dilations,
                                   dtype};

  auto handle = ctx.cudnn_handle();
  auto workspace_handle = ctx.cudnn_workspace_handle();
  paddle::platform::DataLayout layout =
      compute_format == paddle::platform::DataLayout::kNHWC
          ? paddle::platform::DataLayout::kNHWC
          : paddle::platform::DataLayout::kNCHW;
  if (transformed_input.dims().size() == 5) {
    layout = compute_format == paddle::platform::DataLayout::kNHWC
                 ? paddle::platform::DataLayout::kNDHWC
                 : paddle::platform::DataLayout::kNCDHW;
  }
  auto layout_format = paddle::platform::GetCudnnTensorFormat(layout);

  args.handle = handle;

#ifdef PADDLE_WITH_HIP
  // MIOPEN need to set groups in cdesc in miopen_desc.h
  args.cdesc.set(dtype,
                 padding_common,
                 strides,
                 dilations,
                 paddle::platform::AllowTF32Cudnn(),
                 groups);
#else
  args.cdesc.set(dtype,
                 padding_common,
                 strides,
                 dilations,
                 paddle::platform::AllowTF32Cudnn());
#endif

#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION_MIN(7, 0, 1)
  // cudnn 7 can support groups, no need to do it manually
  // FIXME(typhoonzero): find a better way to disable groups
  // rather than setting it to 1.
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cudnnSetConvolutionGroupCount(
          args.cdesc.desc(), groups));
  groups = 1;
#endif
#ifdef PADDLE_WITH_HIP
  // MIOPEN do not set groups in wdesc after set groups in cdesc
  groups = 1;
#endif
  args.idesc.set(transformed_input, layout_format);
  args.wdesc.set(transformed_filter_channel, layout_format, groups);
  args.odesc.set(transformed_output, layout_format);
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
    paddle::operators::GetNCDHW(transformed_output.dims(),
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
    paddle::operators::GetNCDHW(transformed_output.dims(),
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
  // ------------------- cudnn conv workspace ---------------------
  size_t workspace_size = 0;  // final workspace to allocate.
// ------------------- cudnn conv algorithm ---------------------
#ifdef PADDLE_WITH_HIP
  paddle::operators::SearchResult<miopenConvFwdAlgorithm_t> fwd_result;
  using search = paddle::operators::SearchAlgorithm<miopenConvFwdAlgorithm_t>;
  workspace_size = search::GetWorkspaceSize(args);
  fwd_result.algo = search::Find<T>(
      args, exhaustive_search, deterministic, workspace_size, ctx);
#else
  paddle::operators::SearchResult<cudnnConvolutionFwdAlgo_t> fwd_result;
  using search =
      paddle::operators::SearchAlgorithm<cudnnConvolutionFwdAlgoPerf_t>;
  fwd_result = search::Find<T>(args, exhaustive_search, deterministic, ctx);
  workspace_size = search::GetWorkspaceSize(args, fwd_result.algo);
#endif

#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION_MIN(7, 0, 1)
  // when groups > 1, SearchAlgorithm find algo is CUDNN_CONVOLUTION_\
    // FWD_ALGO_WINOGRAD_NONFUSED, but this kind of algorithm is unstable
  // in forward computation, so change the algorithm to CUDNN_CONVOLUTION_\
    // FWD_ALGO_IMPLICIT_GEMM manually.
  if (groups > 1) {
    fwd_result.algo = static_cast<cudnnConvolutionFwdAlgo_t>(0);
  }
#endif

  // ------------------- cudnn conv forward ---------------------
  paddle::operators::ScalingParamType<T> alpha = 1.0f;
  paddle::operators::ScalingParamType<T> beta = 0.0f;

// NOTE(zhiqiu): inplace addto is not supportted in double grad yet.
// ScalingParamType<T> beta = ctx.Attr<bool>("use_addto") ? 1.0f : 0.0f;
// VLOG(4) << "Conv: use_addto = " << ctx.Attr<bool>("use_addto");

#ifdef PADDLE_WITH_HIP
  workspace_handle.RunFunc(
      [&](void* workspace_ptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            paddle::platform::dynload::miopenConvolutionForward(
                handle,
                &alpha,
                args.idesc.desc(),
                input_data,
                args.wdesc.desc(),
                filter_data,
                args.cdesc.desc(),
                fwd_result.algo,
                &beta,
                args.odesc.desc(),
                output_data,
                workspace_ptr,
                workspace_size));
      },
      workspace_size);
#else
  for (int i = 0; i < groups; i++) {
    workspace_handle.RunFunc(
        [&](void* workspace_ptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              paddle::platform::dynload::cudnnConvolutionForward(
                  handle,
                  &alpha,
                  args.idesc.desc(),
                  input_data + i * group_offset_in,
                  args.wdesc.desc(),
                  filter_data + i * group_offset_filter,
                  args.cdesc.desc(),
                  fwd_result.algo,
                  workspace_ptr,
                  workspace_size,
                  &beta,
                  args.odesc.desc(),
                  output_data + i * group_offset_out));
        },
        workspace_size);
  }
#endif

  if (channel_last && compute_format == paddle::platform::DataLayout::kNCHW) {
    TransToChannelLast<Context, T>(ctx, &transformed_output, output);
  }
}

template <typename T, typename Context>
void Conv3DCudnnKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& filter,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::string& padding_algorithm,
                       int groups,
                       const std::vector<int>& dilations,
                       const std::string& data_format,
                       bool use_addto,
                       int workspace_size_MB,
                       bool exhaustive_search,
                       DenseTensor* out) {
  ConvCudnnKernel<T>(dev_ctx,
                     input,
                     filter,
                     strides,
                     paddings,
                     padding_algorithm,
                     groups,
                     dilations,
                     data_format,
                     use_addto,
                     workspace_size_MB,
                     exhaustive_search,
                     out);
}

template <typename T, typename Context>
void DepthwiseConvCudnnKernel(const Context& dev_ctx,
                              const DenseTensor& input,
                              const DenseTensor& filter,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::string& padding_algorithm,
                              int groups,
                              const std::vector<int>& dilations,
                              const std::string& data_format,
                              bool use_addto,
                              int workspace_size_MB,
                              bool exhaustive_search,
                              bool fuse_relu,
                              DenseTensor* out) {
  ConvCudnnKernel<T>(dev_ctx,
                     input,
                     filter,
                     strides,
                     paddings,
                     padding_algorithm,
                     groups,
                     dilations,
                     data_format,
                     use_addto,
                     workspace_size_MB,
                     exhaustive_search,
                     out);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(conv2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(depthwise_conv2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::DepthwiseConvCudnnKernel,
                   float,
                   phi::dtype::float16) {}

#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(conv2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(conv3d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(conv2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif

#endif

// todo register bfloat16
