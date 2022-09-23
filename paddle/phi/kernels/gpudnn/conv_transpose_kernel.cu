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

#include "paddle/phi/kernels/conv_transpose_kernel.h"

#include <algorithm>

#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/padding.h"
#include "paddle/phi/kernels/funcs/slice.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/operators/conv_miopen_helper.h"
#include "paddle/fluid/platform/device/gpu/rocm/miopen_helper.h"
#else
#include "paddle/fluid/operators/conv_cudnn_helper.h"
#include "paddle/fluid/platform/device/gpu/cuda/cudnn_helper.h"
#endif

namespace phi {

using GPUDNNDataLayout = paddle::platform::DataLayout;

template <typename T, typename Context>
void ConvTransposeRawGPUDNNKernel(const Context& ctx,
                                  const DenseTensor& x,
                                  const DenseTensor& filter,
                                  const std::vector<int>& strides,
                                  const std::vector<int>& paddings,
                                  const std::string& padding_algorithm,
                                  int groups,
                                  const std::vector<int>& dilations,
                                  const std::string& data_format,
                                  DenseTensor* out) {
  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ =
      dilations;  // cudnn v5 does not support dilations
  const T* filter_data = filter.data<T>();
  const GPUDNNDataLayout data_layout =
      (data_format != "NHWC" ? GPUDNNDataLayout::kNCHW
                             : GPUDNNDataLayout::kNHWC);
  std::vector<int> x_vec = vectorize<int>(x.dims());
  std::vector<int> out_vec = vectorize<int>(out->dims());
  // if channel_last, transpose to channel_first
  DenseTensor x_transpose;
  if (data_layout == GPUDNNDataLayout::kNHWC) {
    if (strides.size() == 2U) {
      std::vector<int> axis = {0, 3, 1, 2};
      for (size_t i = 0; i < axis.size(); ++i) {
        x_vec[i] = x.dims()[axis[i]];
        out_vec[i] = out->dims()[axis[i]];
      }
      x_transpose = Transpose<T, Context>(ctx, x, axis);
    } else if (strides.size() == 3U) {
      std::vector<int> axis = {0, 4, 1, 2, 3};
      for (size_t i = 0; i < axis.size(); ++i) {
        x_vec[i] = x.dims()[axis[i]];
        out_vec[i] = out->dims()[axis[i]];
      }
      x_transpose = Transpose<T, Context>(ctx, x, axis);
    }
  } else {
    x_transpose = x;
  }

  // update padding and dilation
  auto x_dims = x_transpose.dims();
  auto filter_dims = filter.dims();
  DDim x_data_dims;
  x_data_dims = slice_ddim(x_dims, 2, x_dims.size());
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, x_data_dims, strides, ksize);

  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings_, data_dim);

  std::vector<int> x_pad(x_dims.size() * 2, 0);
  DenseTensor transformed_x;
  std::vector<int> padding_common(data_dim, 0);
  if (!is_sys_pad) {
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_x_shape_vec(data_dim + 2);
    new_x_shape_vec[0] = x_dims[0];
    new_x_shape_vec[1] = x_dims[1];

    for (size_t i = 0; i < data_dim; ++i) {
      padding_diff[i] = std::abs(paddings_[2 * i] - paddings_[2 * i + 1]);
      padding_common[i] = std::min(paddings_[2 * i], paddings_[2 * i + 1]);
      new_x_shape_vec[i + 2] = x_dims[i + 2] + padding_diff[i];
      x_pad[2 * i + 4] = paddings_[2 * i] - padding_common[i];
      x_pad[2 * i + 4 + 1] = paddings_[2 * i + 1] - padding_common[i];
    }
    DDim new_x_shape(make_ddim(new_x_shape_vec));
    transformed_x.Resize(new_x_shape);
    ctx.template Alloc<T>(&transformed_x);

    const int rank = x_dims.size();
    T pad_value(0.0);
    switch (rank) {
      case 4: {
        funcs::PadFunction<Context, T, 4>(
            ctx, x_pad, x_transpose, pad_value, &transformed_x);
      } break;
      case 5: {
        funcs::PadFunction<Context, T, 5>(
            ctx, x_pad, x_transpose, pad_value, &transformed_x);
      } break;
      default:
        PADDLE_THROW(errors::InvalidArgument(
            "Op(ConvTranspose) only supports 4-D or 5-D x DenseTensor."));
    }
  } else {
    transformed_x = x_transpose;
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
    starts[i] = x_pad[2 * i + 4] * (strides[i] + 1);
    ends[i] = starts[i] + out_vec[i + 2];
    axes[i] = i + 2;
  }

  const T* x_data = transformed_x.data<T>();
  x_vec = vectorize<int>(transformed_x.dims());

  std::vector<int> transformed_out_vec = out_vec;
  for (size_t i = 0; i < data_dim; ++i) {
    transformed_out_vec[i + 2] =
        out_vec[i + 2] + (x_pad[2 * i + 4] + x_pad[2 * i + 5]) * strides[i] -
        2 * padding_common[i] + paddings_[2 * i] + paddings_[2 * i + 1];
  }

  DenseTensor transformed_out;
  if (!is_sys_pad) {
    transformed_out.Resize(make_ddim(transformed_out_vec));
    ctx.template Alloc<T>(&transformed_out);
  } else {
    ctx.template Alloc<T>(out);
    transformed_out.ShareDataWith(*out);
    transformed_out.Resize(make_ddim(transformed_out_vec));
  }
  T* transformed_out_data = transformed_out.data<T>();

  GPUDNNDataLayout layout;

  int iwo_groups = groups;
  int c_groups = 1;
#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
  iwo_groups = 1;
  c_groups = groups;
  groups = 1;
#endif

  if (strides.size() == 2U) {
    layout = GPUDNNDataLayout::kNCHW;
  } else {
    layout = GPUDNNDataLayout::kNCDHW;
  }

  size_t workspace_size = 0;
#ifdef PADDLE_WITH_HIP
  miopenConvBwdDataAlgorithm_t algo{};
#else
  cudnnConvolutionBwdDataAlgo_t algo{};
#endif
  // ------------------- cudnn conv algorithm ---------------------
  auto handle = ctx.cudnn_handle();
  auto layout_tensor = paddle::platform::GetCudnnTensorFormat(layout);
  bool deterministic = FLAGS_cudnn_deterministic;

  auto dtype = paddle::platform::CudnnDataType<T>::type;
  // ------------------- cudnn descriptors ---------------------
  paddle::operators::ConvArgs args{&transformed_out,
                                   &filter,
                                   &transformed_x,
                                   strides,
                                   padding_common,
                                   dilations_,
                                   dtype,
                                   groups,
                                   data_layout};
  args.handle = handle;
  args.idesc.set(transformed_out, iwo_groups);
  args.wdesc.set(filter, layout_tensor, iwo_groups);
  args.odesc.set(transformed_x, iwo_groups);
  args.cdesc.set(dtype,
                 padding_common,
                 strides,
                 dilations_,
                 paddle::platform::AllowTF32Cudnn(),
                 c_groups);

#ifdef PADDLE_WITH_HIP
  paddle::operators::SearchResult<miopenConvBwdDataAlgorithm_t> bwd_result;
  using search =
      paddle::operators::SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
  workspace_size = std::max(workspace_size, search::GetWorkspaceSize(args));
  bwd_result.algo =
      search::Find<T>(args, false, deterministic, workspace_size, ctx);
#else
  paddle::operators::SearchResult<cudnnConvolutionBwdDataAlgo_t> bwd_result;
  using search =
      paddle::operators::SearchAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>;
  bwd_result = search::Find<T>(args, false, deterministic, ctx);
  workspace_size =
      std::max(workspace_size, search::GetWorkspaceSize(args, bwd_result.algo));
#endif

  // ------------------- cudnn conv transpose forward ---------------------
  int x_offset = transformed_x.numel() / transformed_x.dims()[0] / groups;
  int out_offset = transformed_out.numel() / transformed_out.dims()[0] / groups;
  int filter_offset = filter.numel() / groups;
  paddle::operators::ScalingParamType<T> alpha = 1.0f;
  paddle::operators::ScalingParamType<T> beta = 0.0f;
  auto workspace_handle = ctx.cudnn_workspace_handle();
  for (int g = 0; g < groups; g++) {
#ifdef PADDLE_WITH_HIP
    auto cudnn_func = [&](void* cudnn_workspace) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenConvolutionBackwardData(
          handle,
          &alpha,
          args.odesc.desc(),
          x_data + x_offset * g,
          args.wdesc.desc(),
          filter_data + filter_offset * g,
          args.cdesc.desc(),
          bwd_result.algo,
          &beta,
          args.idesc.desc(),
          transformed_out_data + out_offset * g,
          cudnn_workspace,
          workspace_size));
    };
#else   // PADDLE_WITH_HIP
    auto cudnn_func = [&](void* cudnn_workspace) {
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cudnnConvolutionBackwardData(
          handle,
          &alpha,
          args.wdesc.desc(),
          filter_data + filter_offset * g,
          args.odesc.desc(),
          x_data + x_offset * g,
          args.cdesc.desc(),
          bwd_result.algo,
          cudnn_workspace,
          workspace_size,
          &beta,
          args.idesc.desc(),
          transformed_out_data + out_offset * g));
    };
#endif  // PADDLE_WITH_HIP
    workspace_handle.RunFunc(cudnn_func, workspace_size);
  }
  if (!is_sys_pad && strides.size() == 2U) {
    funcs::Slice<Context, T, 4>(ctx, &transformed_out, out, starts, ends, axes);
  } else if (!is_sys_pad && strides.size() == 3U) {
    funcs::Slice<Context, T, 5>(ctx, &transformed_out, out, starts, ends, axes);
  }

  if (data_layout == GPUDNNDataLayout::kNHWC) {
    DenseTensor out_transpose;
    DenseTensor out_nchw;
    out_nchw.ShareDataWith(*out);
    out_nchw.Resize(make_ddim(out_vec));

    if (strides.size() == 2U) {
      out_transpose = Transpose<T, Context>(ctx, out_nchw, {0, 2, 3, 1});
    } else if (strides.size() == 3U) {
      out_transpose = Transpose<T, Context>(ctx, out_nchw, {0, 2, 3, 4, 1});
    }
    *out = out_transpose;
  }
}

template <typename T, typename Context>
void Conv2dTransposeGPUDNNKernel(const Context& ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& filter,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 const std::vector<int>& output_padding,
                                 const IntArray& output_size,
                                 const std::string& padding_algorithm,
                                 int groups,
                                 const std::vector<int>& dilations,
                                 const std::string& data_format,
                                 DenseTensor* out) {
  ConvTransposeRawGPUDNNKernel<T, Context>(ctx,
                                           x,
                                           filter,
                                           strides,
                                           paddings,
                                           padding_algorithm,
                                           groups,
                                           dilations,
                                           data_format,
                                           out);
}

template <typename T, typename Context>
void Conv3dTransposeGPUDNNKernel(const Context& ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& filter,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 const std::vector<int>& output_padding,
                                 const std::vector<int>& output_size,
                                 const std::string& padding_algorithm,
                                 int groups,
                                 const std::vector<int>& dilations,
                                 const std::string& data_format,
                                 DenseTensor* out) {
  ConvTransposeRawGPUDNNKernel<T, Context>(ctx,
                                           x,
                                           filter,
                                           strides,
                                           paddings,
                                           padding_algorithm,
                                           groups,
                                           dilations,
                                           data_format,
                                           out);
}

}  // namespace phi

using float16 = phi::dtype::float16;

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(conv2d_transpose,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeGPUDNNKernel,
                   float,
                   float16) {}
PD_REGISTER_KERNEL(conv3d_transpose,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3dTransposeGPUDNNKernel,
                   float,
                   float16) {}
#else
PD_REGISTER_KERNEL(conv2d_transpose,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeGPUDNNKernel,
                   float,
                   double,
                   float16) {}
PD_REGISTER_KERNEL(conv3d_transpose,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3dTransposeGPUDNNKernel,
                   float,
                   double,
                   float16) {}
#endif
