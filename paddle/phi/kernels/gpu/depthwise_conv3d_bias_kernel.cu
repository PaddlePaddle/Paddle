// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/gpu/depthwise_conv.h"

namespace phi {

template <typename T,
          typename AccT,
          int kKnownKernelT,
          int kKnownKernelH,
          int kKnownKernelW,
          int kKnownDilationT,
          int kKnownDilationH,
          int kKnownDilationW>
__global__ void DWConv3dFwdKernel(const T* input,
                                  T* output,
                                  const T* weight,
                                  const T* bias,
                                  const int batch_size,
                                  const int input_channels,
                                  const int input_depth,
                                  int input_height,
                                  int input_width,
                                  const int output_channels,
                                  const int output_depth,
                                  int output_height,
                                  int output_width,
                                  const int kernel_t_in,
                                  int kernel_h_in,
                                  int kernel_w_in,
                                  const int stride_t,
                                  int stride_h,
                                  int stride_w,
                                  const int padding_t,
                                  int padding_h,
                                  int padding_w,
                                  const int dilation_t_in,
                                  int dilation_h_in,
                                  int dilation_w_in) {
  const int kernel_t = kKnownKernelT > 0 ? kKnownKernelT : kernel_t_in;
  const int kernel_h = kKnownKernelH > 0 ? kKnownKernelH : kernel_h_in;
  const int kernel_w = kKnownKernelW > 0 ? kKnownKernelW : kernel_w_in;
  const int dilation_t = kKnownDilationT > 0 ? kKnownDilationT : dilation_t_in;
  const int dilation_h = kKnownDilationH > 0 ? kKnownDilationH : dilation_h_in;
  const int dilation_w = kKnownDilationW > 0 ? kKnownDilationW : dilation_w_in;

  const int num_outputs = batch_size * output_channels * output_depth *
                          output_height * output_width;
  const int channel_multiplier = output_channels / input_channels;

  const int i_stride_c = input_depth * input_height * input_width;
  const int i_stride_d = input_height * input_width;
  const int i_stride_h = input_width;

  const int w_stride_c = kernel_t * kernel_h * kernel_w;

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_outputs;
       index += blockDim.x * gridDim.x) {
    int temp = index;
    const int w_out = temp % output_width;
    temp /= output_width;
    const int h_out = temp % output_height;
    temp /= output_height;
    const int d_out = temp % output_depth;
    temp /= output_depth;
    const int c_out = temp % output_channels;
    const int b = temp / output_channels;

    const int c_in = c_out / channel_multiplier;

    const int d_in_start = d_out * stride_t - padding_t;
    const int h_in_start = h_out * stride_h - padding_h;
    const int w_in_start = w_out * stride_w - padding_w;

    AccT sum = 0;
    const T* weight_ptr = weight + c_out * w_stride_c;
    const T* input_base =
        input + b * (input_channels * i_stride_c) + c_in * i_stride_c;

    for (int kt = 0; kt < kernel_t; ++kt) {
      const int d_in = d_in_start + kt * dilation_t;
      for (int kh = 0; kh < kernel_h; ++kh) {
        const int h_in = h_in_start + kh * dilation_h;
        for (int kw = 0; kw < kernel_w; ++kw) {
          const int w_in = w_in_start + kw * dilation_w;

          const T val = *weight_ptr;
          weight_ptr++;

          if (d_in >= 0 && d_in < input_depth && h_in >= 0 &&
              h_in < input_height && w_in >= 0 && w_in < input_width) {
            const int input_offset =
                d_in * i_stride_d + h_in * i_stride_h + w_in;
            sum += static_cast<AccT>(val) *
                   static_cast<AccT>(input_base[input_offset]);
          }
        }
      }
    }

    if (bias != nullptr) {
      sum += static_cast<AccT>(bias[c_out]);
    }

    output[index] = static_cast<T>(sum);
  }
}

template <typename T, typename Context>
void LaunchDepthwiseConv3dCompatible(const Context& dev_ctx,
                                     const DenseTensor& input,
                                     const DenseTensor& filter,
                                     const DenseTensor* bias,
                                     const std::vector<int>& strides,
                                     const std::vector<int>& paddings,
                                     const std::vector<int>& dilations,
                                     const std::string& data_format,
                                     DenseTensor* out) {
  const bool channel_last = (data_format == "NDHWC");

  DenseTensor input_ncdhw;
  DenseTensor out_ncdhw;
  const DenseTensor& filter_ncdhw = filter;

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(dev_ctx, &input, &input_ncdhw);
    TransToChannelFirst<Context, T>(dev_ctx, &input, &input_ncdhw);

    ResizeToChannelFirst<Context, T>(dev_ctx, out, &out_ncdhw);
    dev_ctx.template Alloc<T>(&out_ncdhw);
  } else {
    input_ncdhw.ShareDataWith(input);
    out_ncdhw.ShareDataWith(*out);
  }

  // Get Input and Output dims
  const int64_t batch_size = input_ncdhw.dims()[0];
  const int64_t in_channels = input_ncdhw.dims()[1];
  const int64_t in_depth = input_ncdhw.dims()[2];
  const int64_t in_height = input_ncdhw.dims()[3];
  const int64_t in_width = input_ncdhw.dims()[4];

  const int64_t out_channels = filter_ncdhw.dims()[0];
  const int64_t kernel_t = filter_ncdhw.dims()[2];
  const int64_t kernel_h = filter_ncdhw.dims()[3];
  const int64_t kernel_w = filter_ncdhw.dims()[4];
  std::vector<int> kernel_size = {static_cast<int>(kernel_t),
                                  static_cast<int>(kernel_h),
                                  static_cast<int>(kernel_w)};

  const int64_t out_depth = out_ncdhw.dims()[2];
  const int64_t out_height = out_ncdhw.dims()[3];
  const int64_t out_width = out_ncdhw.dims()[4];

  std::vector<int> paddings_vec;
  if (paddings.size() == 6) {
    paddings_vec = {paddings[0], paddings[2], paddings[4]};
  } else if (paddings.size() == 3) {
    paddings_vec = paddings;
  } else {
    paddings_vec = {paddings[0], paddings[0], paddings[0]};
  }

  const T* bias_ptr = nullptr;
  if (bias && bias->initialized() && bias->numel() > 0) {
    bias_ptr = bias->data<T>();
  }

  int64_t num_outputs = out->numel();
  int block = 256;
  int grid = std::min((num_outputs - 1) / block + 1, (int64_t)65536);
  auto stream = dev_ctx.stream();

  using AccT = typename dtype::MPTypeTrait<T>::Type;

  const T* input_ptr = input_ncdhw.data<T>();
  T* output_ptr = out_ncdhw.data<T>();
  const T* filter_ptr = filter_ncdhw.data<T>();

  bool is_kernel_3x3x3 = (kernel_t == 3 && kernel_h == 3 && kernel_w == 3);
  bool is_dilation_1x1x1 =
      (dilations[0] == 1 && dilations[1] == 1 && dilations[2] == 1);

  if (is_kernel_3x3x3 && is_dilation_1x1x1) {
    DWConv3dFwdKernel<T, AccT, 3, 3, 3, 1, 1, 1>
        <<<grid, block, 0, stream>>>(input_ptr,
                                     output_ptr,
                                     filter_ptr,
                                     bias_ptr,
                                     static_cast<int>(batch_size),
                                     static_cast<int>(in_channels),
                                     static_cast<int>(in_depth),
                                     static_cast<int>(in_height),
                                     static_cast<int>(in_width),
                                     static_cast<int>(out_channels),
                                     static_cast<int>(out_depth),
                                     static_cast<int>(out_height),
                                     static_cast<int>(out_width),
                                     static_cast<int>(kernel_t),
                                     static_cast<int>(kernel_h),
                                     static_cast<int>(kernel_w),
                                     strides[0],
                                     strides[1],
                                     strides[2],
                                     paddings_vec[0],
                                     paddings_vec[1],
                                     paddings_vec[2],
                                     dilations[0],
                                     dilations[1],
                                     dilations[2]);
  } else if (is_dilation_1x1x1) {
    DWConv3dFwdKernel<T, AccT, -1, -1, -1, 1, 1, 1>
        <<<grid, block, 0, stream>>>(input_ptr,
                                     output_ptr,
                                     filter_ptr,
                                     bias_ptr,
                                     static_cast<int>(batch_size),
                                     static_cast<int>(in_channels),
                                     static_cast<int>(in_depth),
                                     static_cast<int>(in_height),
                                     static_cast<int>(in_width),
                                     static_cast<int>(out_channels),
                                     static_cast<int>(out_depth),
                                     static_cast<int>(out_height),
                                     static_cast<int>(out_width),
                                     static_cast<int>(kernel_t),
                                     static_cast<int>(kernel_h),
                                     static_cast<int>(kernel_w),
                                     strides[0],
                                     strides[1],
                                     strides[2],
                                     paddings_vec[0],
                                     paddings_vec[1],
                                     paddings_vec[2],
                                     dilations[0],
                                     dilations[1],
                                     dilations[2]);
  } else {
    DWConv3dFwdKernel<T, AccT, -1, -1, -1, -1, -1, -1>
        <<<grid, block, 0, stream>>>(input_ptr,
                                     output_ptr,
                                     filter_ptr,
                                     bias_ptr,
                                     static_cast<int>(batch_size),
                                     static_cast<int>(in_channels),
                                     static_cast<int>(in_depth),
                                     static_cast<int>(in_height),
                                     static_cast<int>(in_width),
                                     static_cast<int>(out_channels),
                                     static_cast<int>(out_depth),
                                     static_cast<int>(out_height),
                                     static_cast<int>(out_width),
                                     static_cast<int>(kernel_t),
                                     static_cast<int>(kernel_h),
                                     static_cast<int>(kernel_w),
                                     strides[0],
                                     strides[1],
                                     strides[2],
                                     paddings_vec[0],
                                     paddings_vec[1],
                                     paddings_vec[2],
                                     dilations[0],
                                     dilations[1],
                                     dilations[2]);
  }

  if (channel_last) {
    TransToChannelLast<Context, T>(dev_ctx, &out_ncdhw, out);
  }
}

template <typename T, typename Context>
void DepthwiseConv3dBiasKernel(const Context& dev_ctx,
                               const DenseTensor& input,
                               const DenseTensor& filter,
                               const optional<DenseTensor>& bias,
                               const std::vector<int>& strides_t,
                               const std::vector<int>& paddings_t,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations_t,
                               const std::string& data_format,
                               DenseTensor* out) {
  // Check & Alloc (0-size)
  if (input.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }

  DenseTensor* output = out;
  dev_ctx.template Alloc<T>(output);

  std::vector<int> strides = strides_t;
  std::vector<int> dilations = dilations_t;
  std::vector<int> paddings = paddings_t;

  // Channel Check
  const bool channel_last = (data_format == "NDHWC");
  int c_in_idx = channel_last ? 4 : 1;
  int c_out_idx = channel_last ? 4 : 1;

  PADDLE_ENFORCE_EQ(output->dims()[c_out_idx] % input.dims()[c_in_idx],
                    0,
                    common::errors::InvalidArgument(
                        "Output channels must be multiple of input channels"));

  // Update Padding
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();

  DDim in_data_dims;
  const DataLayout data_layout = StringToDataLayout(data_format);
  if (data_layout != DataLayout::NDHWC) {
    in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  } else {
    in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
  }

  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  // [Front, Back, Top, Bottom, Left, Right] -> [Front, Top, Left]
  bool is_sys_pad = strides.size() * 2 == paddings.size() ? false : true;
  if (!is_sys_pad) {
    for (size_t i = 0; i < strides.size(); ++i) {
      paddings.erase(paddings.begin() + i + 1);
    }
  }

  LaunchDepthwiseConv3dCompatible<T, Context>(dev_ctx,
                                              input,
                                              filter,
                                              bias.get_ptr(),
                                              strides,
                                              paddings,
                                              dilations,
                                              data_format,
                                              out);
}

}  // namespace phi

PD_REGISTER_KERNEL(depthwise_conv3d_bias,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv3dBiasKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
