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

#include "paddle/common/enforce.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/gpu/depthwise_conv.h"

namespace phi {

constexpr int CUDA_NUM_THREADS = 1024;

template <typename Context>
inline uint32_t GET_BLOCKS(
    const Context& dev_ctx,
    const int64_t N,
    const int64_t max_threads_per_block = CUDA_NUM_THREADS) {
  const int64_t block_num = (N - 1) / max_threads_per_block + 1;
  PADDLE_ENFORCE_LE_UINT32_MAX(block_num, "depthwise conv2d bias grid.x");
  PADDLE_ENFORCE_LE(block_num,
                    static_cast<int64_t>(dev_ctx.GetCUDAMaxGridDimSize()[0]),
                    common::errors::InvalidArgument(
                        "depthwise conv2d bias grid.x exceeds device limit."));
  return static_cast<uint32_t>(block_num);
}

template <int kSize, typename T, typename IndexT>
__global__ void DWConv2dFwdKernel(const T* __restrict__ input,
                                  T* __restrict__ output,
                                  const T* __restrict__ weight,
                                  const T* __restrict__ bias,
                                  bool biasEnabled,
                                  IndexT totalElements,
                                  const int outputChannels,
                                  const int depthwiseMultiplier,
                                  const int inputWidth,
                                  const int inputHeight,
                                  const int outputWidth,
                                  const int outputHeight,
                                  const int kernelWidth,
                                  const int kernelHeight,
                                  const int strideWidth,
                                  const int strideHeight,
                                  const int padWidth,
                                  const int padHeight,
                                  const int dilationWidth,
                                  const int dilationHeight) {
  using AccT = typename MPTypeTrait<T>::Type;
  const int KW_LIMIT = (kSize != 0) ? kSize : kernelWidth;
  const int KH_LIMIT = (kSize != 0) ? kSize : kernelHeight;

  for (int64_t linearIndex =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t indtmp1 = static_cast<int64_t>(linearIndex) / outputWidth;
    const int64_t w = static_cast<int64_t>(linearIndex) - indtmp1 * outputWidth;
    int64_t indtmp2 = indtmp1 / outputHeight;
    const int64_t h = indtmp1 - indtmp2 * outputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / outputChannels;
    const int64_t c = indtmp1 - indtmp2 * outputChannels;
    const int64_t n = indtmp2;

    int64_t inputChannel = c;
    int64_t inputChannels = outputChannels;
    if (depthwiseMultiplier != 1) {
      inputChannel /= depthwiseMultiplier;
      inputChannels /= depthwiseMultiplier;
    }

    int64_t weightOffset = static_cast<int64_t>(c) * kernelHeight * kernelWidth;
    AccT value = biasEnabled ? static_cast<AccT>(bias[c]) : AccT(0);
    const int64_t offset0 =
        (static_cast<int64_t>(n) * inputChannels + inputChannel) * inputHeight *
        inputWidth;

    for (int kH = 0; kH < KH_LIMIT; ++kH) {
      for (int kW = 0; kW < KW_LIMIT; ++kW) {
        const int64_t h_in = -static_cast<int64_t>(padHeight) +
                             static_cast<int64_t>(h) * strideHeight +
                             static_cast<int64_t>(kH) * dilationHeight;
        const int64_t w_in = -static_cast<int64_t>(padWidth) +
                             static_cast<int64_t>(w) * strideWidth +
                             static_cast<int64_t>(kW) * dilationWidth;

        if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) &&
            (w_in < inputWidth)) {
          const int64_t offset =
              offset0 + static_cast<int64_t>(h_in) * inputWidth + w_in;
          value += (static_cast<AccT>(weight[weightOffset]) *
                    static_cast<AccT>(input[offset]));
        }
        ++weightOffset;
      }
    }
    output[linearIndex] = static_cast<T>(value);
  }
}

template <typename T, typename IndexT>
__global__ void DWConv2dFwdKernelGeneric(const T* __restrict__ input,
                                         T* __restrict__ output,
                                         const T* __restrict__ weight,
                                         const T* __restrict__ bias,
                                         bool biasEnabled,
                                         IndexT totalElements,
                                         const int outputChannels,
                                         const int depthwiseMultiplier,
                                         const int inputWidth,
                                         const int inputHeight,
                                         const int outputWidth,
                                         const int outputHeight,
                                         const int kernelWidth,
                                         const int kernelHeight,
                                         const int strideWidth,
                                         const int strideHeight,
                                         const int padWidth,
                                         const int padHeight,
                                         const int dilationWidth,
                                         const int dilationHeight) {
  using AccT = typename MPTypeTrait<T>::Type;

  for (int64_t linearIndex =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t indtmp1 = static_cast<int64_t>(linearIndex) / outputWidth;
    const int64_t w = static_cast<int64_t>(linearIndex) - indtmp1 * outputWidth;
    int64_t indtmp2 = indtmp1 / outputHeight;
    const int64_t h = indtmp1 - indtmp2 * outputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / outputChannels;
    const int64_t c = indtmp1 - indtmp2 * outputChannels;
    const int64_t n = indtmp2;

    int64_t inputChannel = c;
    int64_t inputChannels = outputChannels;
    if (depthwiseMultiplier != 1) {
      inputChannel /= depthwiseMultiplier;
      inputChannels /= depthwiseMultiplier;
    }

    int64_t weightOffset = static_cast<int64_t>(c) * kernelHeight * kernelWidth;
    int kHmin = 0, kHmax = kernelHeight, kWmin = 0, kWmax = kernelWidth;

    int64_t h_in_min = -static_cast<int64_t>(padHeight) +
                       static_cast<int64_t>(h) * strideHeight;
    if (h_in_min < 0) {
      kHmin = -h_in_min / dilationHeight;
      if ((-h_in_min) % dilationHeight > 0) kHmin++;
    }
    int64_t h_in_max = h_in_min +
                       static_cast<int64_t>(kernelHeight - 1) * dilationHeight -
                       inputHeight + 1;
    if (h_in_max >= 0) {
      kHmax = kernelHeight - h_in_max / dilationHeight;
      if (h_in_max % dilationHeight > 0) kHmax--;
    }
    int64_t w_in_min =
        -static_cast<int64_t>(padWidth) + static_cast<int64_t>(w) * strideWidth;
    if (w_in_min < 0) {
      kWmin = -w_in_min / dilationWidth;
      if ((-w_in_min) % dilationWidth > 0) kWmin++;
    }
    int64_t w_in_max = w_in_min +
                       static_cast<int64_t>(kernelWidth - 1) * dilationWidth -
                       inputWidth + 1;
    if (w_in_max >= 0) {
      kWmax = kernelWidth - w_in_max / dilationWidth;
      if (w_in_max % dilationWidth > 0) kWmax--;
    }

    AccT value = biasEnabled ? static_cast<AccT>(bias[c]) : AccT(0);
    const int64_t offset0 =
        (static_cast<int64_t>(n) * inputChannels + inputChannel) * inputHeight *
        inputWidth;

    for (int kH = kHmin; kH < kHmax; ++kH) {
      const int64_t h_in = -static_cast<int64_t>(padHeight) +
                           static_cast<int64_t>(h) * strideHeight +
                           static_cast<int64_t>(kH) * dilationHeight;
      for (int kW = kWmin; kW < kWmax; ++kW) {
        const int64_t w_in = -static_cast<int64_t>(padWidth) +
                             static_cast<int64_t>(w) * strideWidth +
                             static_cast<int64_t>(kW) * dilationWidth;
        const int64_t offset =
            offset0 + static_cast<int64_t>(h_in) * inputWidth + w_in;
        const int64_t kernel_offset =
            weightOffset + static_cast<int64_t>(kH) * kernelWidth + kW;
        value += (static_cast<AccT>(weight[kernel_offset]) *
                  static_cast<AccT>(input[offset]));
      }
    }
    output[linearIndex] = static_cast<T>(value);
  }
}

template <typename T, typename Context>
void LaunchDepthwiseConv2dCompatible(const Context& dev_ctx,
                                     const DenseTensor& input,
                                     const DenseTensor& filter,
                                     const DenseTensor* bias,
                                     const std::vector<int>& strides,
                                     const std::vector<int>& paddings,
                                     const std::vector<int>& dilations,
                                     const std::string& data_format,
                                     DenseTensor* out) {
  const bool channel_last = (data_format == "NHWC");

  DenseTensor input_nchw;
  DenseTensor out_nchw;
  const DenseTensor& filter_nchw = filter;

  // Layout Transpose (Input/Output only)
  if (channel_last) {
    ResizeToChannelFirst<Context, T>(dev_ctx, &input, &input_nchw);
    TransToChannelFirst<Context, T>(dev_ctx, &input, &input_nchw);
    ResizeToChannelFirst<Context, T>(dev_ctx, out, &out_nchw);
    dev_ctx.template Alloc<T>(&out_nchw);
  } else {
    input_nchw.ShareDataWith(input);
    out_nchw.ShareDataWith(*out);
  }

  // Extract Params
  int64_t c_in = input_nchw.dims()[1];
  int64_t h_in = input_nchw.dims()[2];
  int64_t w_in = input_nchw.dims()[3];

  int64_t outputChannels = out_nchw.dims()[1];
  int64_t h_out = out_nchw.dims()[2];
  int64_t w_out = out_nchw.dims()[3];

  int64_t kH = filter_nchw.dims()[2];
  int64_t kW = filter_nchw.dims()[3];
  int64_t depthwiseMultiplier = outputChannels / c_in;

  int padH = paddings[0];
  int padW = (paddings.size() == 4) ? paddings[2] : paddings[1];

  int dH = dilations[0];
  int dW = dilations[1];
  int strideH = strides[0];
  int strideW = strides[1];

  PADDLE_ENFORCE_LE_INT_MAX(outputChannels, "outputChannels");
  PADDLE_ENFORCE_LE_INT_MAX(depthwiseMultiplier, "depthwiseMultiplier");
  PADDLE_ENFORCE_LE_INT_MAX(w_in, "w_in");
  PADDLE_ENFORCE_LE_INT_MAX(h_in, "h_in");
  PADDLE_ENFORCE_LE_INT_MAX(w_out, "w_out");
  PADDLE_ENFORCE_LE_INT_MAX(h_out, "h_out");
  PADDLE_ENFORCE_LE_INT_MAX(kW, "kW");
  PADDLE_ENFORCE_LE_INT_MAX(kH, "kH");

  const int outputChannels_int = static_cast<int>(outputChannels);
  const int depthwiseMultiplier_int = static_cast<int>(depthwiseMultiplier);
  const int w_in_int = static_cast<int>(w_in);
  const int h_in_int = static_cast<int>(h_in);
  const int w_out_int = static_cast<int>(w_out);
  const int h_out_int = static_cast<int>(h_out);
  const int kW_int = static_cast<int>(kW);
  const int kH_int = static_cast<int>(kH);

  // Launch Kernel
  int64_t totalElements = out_nchw.numel();
  uint32_t blocks = GET_BLOCKS(dev_ctx, totalElements);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);
  auto stream = dev_ctx.stream();

  const T* input_ptr = input_nchw.data<T>();
  T* output_ptr = out_nchw.data<T>();
  const T* weight_ptr = filter_nchw.data<T>();

  // Add Bias
  const T* bias_ptr = nullptr;
  bool has_bias = false;
  if (bias && bias->initialized() && bias->numel() > 0) {
    bias_ptr = bias->data<T>();
    has_bias = true;
  }

  if (kW == 3 && kH == 3) {
    DWConv2dFwdKernel<3, T, int64_t>
        <<<grid, block, 0, stream>>>(input_ptr,
                                     output_ptr,
                                     weight_ptr,
                                     bias_ptr,
                                     has_bias,
                                     totalElements,
                                     outputChannels_int,
                                     depthwiseMultiplier_int,
                                     w_in_int,
                                     h_in_int,
                                     w_out_int,
                                     h_out_int,
                                     kW_int,
                                     kH_int,
                                     strideW,
                                     strideH,
                                     padW,
                                     padH,
                                     dW,
                                     dH);
  } else if (kW == 1 && kH == 1) {
    DWConv2dFwdKernel<1, T, int64_t>
        <<<grid, block, 0, stream>>>(input_ptr,
                                     output_ptr,
                                     weight_ptr,
                                     bias_ptr,
                                     has_bias,
                                     totalElements,
                                     outputChannels_int,
                                     depthwiseMultiplier_int,
                                     w_in_int,
                                     h_in_int,
                                     w_out_int,
                                     h_out_int,
                                     kW_int,
                                     kH_int,
                                     strideW,
                                     strideH,
                                     padW,
                                     padH,
                                     dW,
                                     dH);
  } else if (kW == 5 && kH == 5) {
    DWConv2dFwdKernel<5, T, int64_t>
        <<<grid, block, 0, stream>>>(input_ptr,
                                     output_ptr,
                                     weight_ptr,
                                     bias_ptr,
                                     has_bias,
                                     totalElements,
                                     outputChannels_int,
                                     depthwiseMultiplier_int,
                                     w_in_int,
                                     h_in_int,
                                     w_out_int,
                                     h_out_int,
                                     kW_int,
                                     kH_int,
                                     strideW,
                                     strideH,
                                     padW,
                                     padH,
                                     dW,
                                     dH);
  } else {
    DWConv2dFwdKernelGeneric<T, int64_t>
        <<<grid, block, 0, stream>>>(input_ptr,
                                     output_ptr,
                                     weight_ptr,
                                     bias_ptr,
                                     has_bias,
                                     totalElements,
                                     outputChannels_int,
                                     depthwiseMultiplier_int,
                                     w_in_int,
                                     h_in_int,
                                     w_out_int,
                                     h_out_int,
                                     kW_int,
                                     kH_int,
                                     strideW,
                                     strideH,
                                     padW,
                                     padH,
                                     dW,
                                     dH);
  }

  if (channel_last) {
    TransToChannelLast<Context, T>(dev_ctx, &out_nchw, out);
  }
}

template <typename T, typename Context>
void DepthwiseConv2dBiasKernel(const Context& dev_ctx,
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
  // Check & Alloc
  if (input.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }
  DenseTensor* output = out;
  dev_ctx.template Alloc<T>(output);

  const std::vector<int> strides = strides_t;
  std::vector<int> dilations = dilations_t;
  std::vector<int> paddings = paddings_t;

  const bool channel_last = (data_format == "NHWC");
  int c_in_idx = channel_last ? input.dims().size() - 1 : 1;
  int c_out_idx = channel_last ? output->dims().size() - 1 : 1;
  PADDLE_ENFORCE_EQ(output->dims()[c_out_idx] % input.dims()[c_in_idx],
                    0,
                    common::errors::InvalidArgument(
                        "Output channels must be multiple of input channels"));

  // Update Padding
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();

  DDim in_data_dims;
  const DataLayout data_layout = StringToDataLayout(data_format);
  if (data_layout != DataLayout::NHWC) {
    in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  } else {
    in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
  }

  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  //  Handle Symmetric Padding [Top, Bottom, Left, Right] -> [Top, Left]
  bool is_sys_pad = strides.size() * 2 == paddings.size() ? false : true;
  if (!is_sys_pad) {
    for (size_t i = 0; i < strides.size(); ++i) {
      paddings.erase(paddings.begin() + i + 1);
    }
  }

  // Launch Compatible Kernel with Bias
  LaunchDepthwiseConv2dCompatible<T, Context>(dev_ctx,
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

PD_REGISTER_KERNEL(depthwise_conv2d_bias,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv2dBiasKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
