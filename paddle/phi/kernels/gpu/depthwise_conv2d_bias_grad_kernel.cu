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

#include <cstdint>
#include "paddle/common/layout.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/depthwise_conv.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

constexpr int CUDA_NUM_THREADS = 1024;
constexpr int CUDA_WARP_SIZE = 32;

inline int GET_BLOCKS(const int64_t N,
                      const int64_t max_threads_per_block = CUDA_NUM_THREADS) {
  int64_t block_num = (N - 1) / max_threads_per_block + 1;
  return static_cast<int>(block_num);
}

inline int GetGradParamsNumThreads(int batchSize) {
  constexpr int MAX_BLOCK_SIZE = 256;
  return std::min(batchSize * CUDA_WARP_SIZE, MAX_BLOCK_SIZE);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value,
                                            unsigned int delta,
                                            int width = CUDA_WARP_SIZE,
                                            unsigned int mask = 0xffffffff) {
#ifdef PADDLE_WITH_HIP
  return __shfl_down(value, delta, width);
#else
  return __shfl_down_sync(mask, value, delta, width);
#endif
}

template <>
__device__ __forceinline__ phi::dtype::float16
WARP_SHFL_DOWN<phi::dtype::float16>(phi::dtype::float16 value,
                                    unsigned int delta,
                                    int width,
                                    unsigned int mask) {
  uint16_t val_as_ushort = *reinterpret_cast<uint16_t*>(&value);
  uint16_t shuffled =
      WARP_SHFL_DOWN<uint16_t>(val_as_ushort, delta, width, mask);
  return *reinterpret_cast<phi::dtype::float16*>(&shuffled);
}

template <>
__device__ __forceinline__ phi::dtype::bfloat16
WARP_SHFL_DOWN<phi::dtype::bfloat16>(phi::dtype::bfloat16 value,
                                     unsigned int delta,
                                     int width,
                                     unsigned int mask) {
  uint16_t val_as_ushort = *reinterpret_cast<uint16_t*>(&value);
  uint16_t shuffled =
      WARP_SHFL_DOWN<uint16_t>(val_as_ushort, delta, width, mask);
  return *reinterpret_cast<phi::dtype::bfloat16*>(&shuffled);
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (CUDA_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceSum(T val, T* shared) {
  const int tid = threadIdx.x;
  const int lid = tid % CUDA_WARP_SIZE;
  const int wid = tid / CUDA_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < (blockDim.x / CUDA_WARP_SIZE)) ? shared[lid] : static_cast<T>(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

template <int kSize, int stride, typename T, typename IndexT>
__global__ void DWConv2dBwdInputKernel(const T* __restrict__ grad_output,
                                       T* __restrict__ grad_input,
                                       const T* __restrict__ weight,
                                       IndexT totalElements,
                                       const int inputChannels,
                                       const int depthwiseMultiplier,
                                       const int outputChannels,
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
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  const int KW_LIMIT = (kSize != 0) ? kSize : kernelWidth;
  const int KH_LIMIT = (kSize != 0) ? kSize : kernelHeight;
  const int strideW = (stride != 0) ? stride : strideWidth;
  const int strideH = (stride != 0) ? stride : strideHeight;

  for (IndexT linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += blockDim.x * gridDim.x) {
    int indtmp1 = linearIndex / inputWidth;
    const int w = linearIndex - indtmp1 * inputWidth;
    int indtmp2 = indtmp1 / inputHeight;
    const int h = indtmp1 - indtmp2 * inputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / inputChannels;
    const int c = indtmp1 - indtmp2 * inputChannels;
    const int n = indtmp2;

    AccT value(0);

    for (int multiplier = 0; multiplier < depthwiseMultiplier; ++multiplier) {
      int och = (c * depthwiseMultiplier) + multiplier;
      int weightOffset = och * kernelHeight * kernelWidth;
      for (int kh = 0; kh < KH_LIMIT; ++kh) {
        for (int kw = 0; kw < KW_LIMIT; ++kw) {
          int h_out = h + padHeight - kh * dilationHeight;
          int w_out = w + padWidth - kw * dilationWidth;

          if ((h_out % strideH == 0) && (w_out % strideW == 0)) {
            h_out = h_out / strideH;
            w_out = w_out / strideW;

            if ((h_out >= 0) && (h_out < outputHeight) && (w_out >= 0) &&
                (w_out < outputWidth)) {
              const int offset =
                  ((n * outputChannels + och) * outputHeight + h_out) *
                      outputWidth +
                  w_out;
              value += (static_cast<AccT>(weight[weightOffset]) *
                        static_cast<AccT>(grad_output[offset]));
            }
          }
          ++weightOffset;
        }
      }
    }
    grad_input[linearIndex] = static_cast<T>(value);
  }
}

template <typename T, typename IndexT>
__global__ void DWConv2dBwdWeightKernel(const T* __restrict__ grad_output,
                                        const T* __restrict__ input,
                                        T* __restrict__ grad_weight,
                                        const int batchSize,
                                        const int inputChannels,
                                        const int kernelChannels,
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
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  const int channelStride = kernelWidth * kernelHeight;

  int bidx = blockIdx.x;
  int kW = bidx % kernelWidth;
  int kH = (bidx / kernelWidth) % kernelHeight;
  int ch = (bidx / channelStride);

  int inputCh = ch / depthwiseMultiplier;

  AccT grad(0);

  const int laneId = threadIdx.x % CUDA_WARP_SIZE;
  const int batch = threadIdx.x / CUDA_WARP_SIZE;
  const int nwarps = blockDim.x / CUDA_WARP_SIZE;
  const int imageElements = outputWidth * outputHeight;

  for (int batchIdx = batch; batchIdx < batchSize; batchIdx += nwarps) {
    for (IndexT idx = laneId; idx < imageElements; idx += CUDA_WARP_SIZE) {
      int go_w_offset = idx % outputWidth;
      int go_h_offset = (idx / outputWidth);

      int i_w_offset =
          (go_w_offset * strideWidth) + (kW * dilationWidth) - padWidth;
      int i_h_offset =
          (go_h_offset * strideHeight) + (kH * dilationHeight) - padHeight;

      if (i_w_offset >= 0 && i_h_offset >= 0 && i_w_offset < inputWidth &&
          i_h_offset < inputHeight) {
        int inputOffset =
            ((batchIdx * inputChannels + inputCh) * inputHeight + i_h_offset) *
                inputWidth +
            i_w_offset;
        int outputOffset =
            ((batchIdx * kernelChannels + ch) * outputHeight) * outputWidth +
            idx;

        grad += (static_cast<AccT>(input[inputOffset]) *
                 static_cast<AccT>(grad_output[outputOffset]));
      }
    }
  }

  extern __shared__ char smem[];
  AccT* buf = reinterpret_cast<AccT*>(smem);
  AccT tval = BlockReduceSum(grad, buf);

  if (threadIdx.x == 0) {
    int weightOffset =
        kW + (kernelWidth * kH) + (kernelWidth * kernelHeight * ch);
    grad_weight[weightOffset] = static_cast<T>(tval);
  }
}

template <typename T, typename Context>
void LaunchDepthwiseConv2dBackwardCompatible(const Context& dev_ctx,
                                             const DenseTensor& input,
                                             const DenseTensor& filter,
                                             const DenseTensor& out_grad,
                                             const std::vector<int>& strides,
                                             const std::vector<int>& paddings,
                                             const std::vector<int>& dilations,
                                             const std::string& data_format,
                                             DenseTensor* input_grad,
                                             DenseTensor* filter_grad,
                                             DenseTensor* bias_grad) {
  const bool channel_last = (data_format == "NHWC");

  DenseTensor input_nchw, filter_nchw, out_grad_nchw;
  DenseTensor* input_grad_nchw_ptr = nullptr;
  DenseTensor* filter_grad_nchw_ptr = nullptr;
  DenseTensor input_grad_tmp;

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(dev_ctx, &input, &input_nchw);
    TransToChannelFirst<Context, T>(dev_ctx, &input, &input_nchw);

    ResizeToChannelFirst<Context, T>(dev_ctx, &out_grad, &out_grad_nchw);
    TransToChannelFirst<Context, T>(dev_ctx, &out_grad, &out_grad_nchw);

    if (input_grad) {
      ResizeToChannelFirst<Context, T>(dev_ctx, input_grad, &input_grad_tmp);
      dev_ctx.template Alloc<T>(&input_grad_tmp);
      input_grad_nchw_ptr = &input_grad_tmp;
    }
  } else {
    input_nchw.ShareDataWith(input);
    out_grad_nchw.ShareDataWith(out_grad);
    if (input_grad) input_grad_nchw_ptr = input_grad;
  }

  filter_nchw.ShareDataWith(filter);

  if (filter_grad) {
    if (channel_last) dev_ctx.template Alloc<T>(filter_grad);
    filter_grad_nchw_ptr = filter_grad;
  }

  int64_t batchSize = input_nchw.dims()[0];
  int64_t c_in = input_nchw.dims()[1];
  int64_t h_in = input_nchw.dims()[2];
  int64_t w_in = input_nchw.dims()[3];

  int64_t outputChannels = out_grad_nchw.dims()[1];
  int64_t h_out = out_grad_nchw.dims()[2];
  int64_t w_out = out_grad_nchw.dims()[3];

  int64_t kH = filter_nchw.dims()[2];
  int64_t kW = filter_nchw.dims()[3];
  int64_t depthwiseMultiplier = outputChannels / c_in;

  int padH = paddings[0];
  int padW = (paddings.size() == 4) ? paddings[2] : paddings[1];
  int dH = dilations[0];
  int dW = dilations[1];
  int strideH = strides[0];
  int strideW = strides[1];

  auto stream = dev_ctx.stream();

  // Launch Filter Gradient Kernel (grad_weight)
  if (filter_grad_nchw_ptr) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, filter_grad_nchw_ptr, static_cast<T>(0));

    int blocks = outputChannels * kH * kW;
    dim3 grid(blocks);
    dim3 block(GetGradParamsNumThreads(batchSize));

    size_t smem = (block.x / CUDA_WARP_SIZE) *
                  sizeof(typename phi::dtype::MPTypeTrait<T>::Type);

    DWConv2dBwdWeightKernel<T, int>
        <<<grid, block, smem, stream>>>(out_grad_nchw.data<T>(),
                                        input_nchw.data<T>(),
                                        filter_grad_nchw_ptr->data<T>(),
                                        static_cast<int>(batchSize),
                                        static_cast<int>(c_in),
                                        static_cast<int>(outputChannels),
                                        static_cast<int>(depthwiseMultiplier),
                                        static_cast<int>(w_in),
                                        static_cast<int>(h_in),
                                        static_cast<int>(w_out),
                                        static_cast<int>(h_out),
                                        static_cast<int>(kW),
                                        static_cast<int>(kH),
                                        strideW,
                                        strideH,
                                        padW,
                                        padH,
                                        dW,
                                        dH);
  }

  // Launch Input Gradient Kernel (grad_input)
  if (input_grad_nchw_ptr) {
    int64_t totalElements = input_grad_nchw_ptr->numel();
    int blocks = GET_BLOCKS(totalElements);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS);

    const T* grad_output_ptr = out_grad_nchw.data<T>();
    T* grad_input_ptr = input_grad_nchw_ptr->data<T>();
    const T* weight_ptr = filter_nchw.data<T>();

#define LAUNCH_INPUT_KERNEL(K, S)                                         \
  DWConv2dBwdInputKernel<K, S, T, int>                                    \
      <<<grid, block, 0, stream>>>(grad_output_ptr,                       \
                                   grad_input_ptr,                        \
                                   weight_ptr,                            \
                                   static_cast<int>(totalElements),       \
                                   static_cast<int>(c_in),                \
                                   static_cast<int>(depthwiseMultiplier), \
                                   static_cast<int>(outputChannels),      \
                                   static_cast<int>(w_in),                \
                                   static_cast<int>(h_in),                \
                                   static_cast<int>(w_out),               \
                                   static_cast<int>(h_out),               \
                                   static_cast<int>(kW),                  \
                                   static_cast<int>(kH),                  \
                                   strideW,                               \
                                   strideH,                               \
                                   padW,                                  \
                                   padH,                                  \
                                   dW,                                    \
                                   dH);

    if (kW == 5 && kH == 5) {
      if (dW == 1 && dH == 1)
        LAUNCH_INPUT_KERNEL(5, 1)
      else if (dW == 2 && dH == 2)
        LAUNCH_INPUT_KERNEL(5, 2)
      else
        LAUNCH_INPUT_KERNEL(5, 0)
    } else if (kW == 3 && kH == 3) {
      if (dW == 1 && dH == 1)
        LAUNCH_INPUT_KERNEL(3, 1)
      else if (dW == 2 && dH == 2)
        LAUNCH_INPUT_KERNEL(3, 2)
      else
        LAUNCH_INPUT_KERNEL(3, 0)
    } else if (kW == 1 && kH == 1) {
      if (dW == 1 && dH == 1)
        LAUNCH_INPUT_KERNEL(1, 1)
      else if (dW == 2 && dH == 2)
        LAUNCH_INPUT_KERNEL(1, 2)
      else
        LAUNCH_INPUT_KERNEL(1, 0)
    } else {
      if (dW == 1 && dH == 1)
        LAUNCH_INPUT_KERNEL(0, 1)
      else if (dW == 2 && dH == 2)
        LAUNCH_INPUT_KERNEL(0, 2)
      else
        LAUNCH_INPUT_KERNEL(0, 0)
    }
#undef LAUNCH_INPUT_KERNEL
  }

  // Bias Gradient
  if (bias_grad) {
    dev_ctx.template Alloc<T>(bias_grad);

    // Reduce over N(0), H(2), W(3) to get [C]
    std::vector<int64_t> reduce_dims = {0, 2, 3};

    phi::SumKernel<T, Context>(dev_ctx,
                               out_grad_nchw,
                               phi::IntArray(reduce_dims),
                               phi::CppTypeToDataType<T>::Type(),
                               false,
                               bias_grad);
  }

  if (input_grad && channel_last) {
    TransToChannelLast<Context, T>(dev_ctx, input_grad_nchw_ptr, input_grad);
  }
}

template <typename T, typename Context>
void DepthwiseConv2dBiasGradKernel(const Context& dev_ctx,
                                   const DenseTensor& input,
                                   const DenseTensor& filter,
                                   const paddle::optional<DenseTensor>& bias,
                                   const DenseTensor& out_grad,
                                   const std::vector<int>& strides_t,
                                   const std::vector<int>& paddings_t,
                                   const std::string& padding_algorithm,
                                   int groups,
                                   const std::vector<int>& dilations_t,
                                   const std::string& data_format,
                                   DenseTensor* input_grad,
                                   DenseTensor* filter_grad,
                                   DenseTensor* bias_grad) {
  const DenseTensor* output_grad = &out_grad;

  if (!input_grad && !filter_grad && !bias_grad) return;
  // 0-size
  if (input.numel() == 0) {
    if (input_grad) dev_ctx.template Alloc<T>(input_grad);
    if (filter_grad) {
      Full<T, Context>(dev_ctx, filter_grad->dims(), 0, filter_grad);
    }
    if (bias_grad) {
      dev_ctx.template Alloc<T>(bias_grad);
      Full<T, Context>(dev_ctx, bias_grad->dims(), 0, bias_grad);
    }
    return;
  }

  if (input_grad) dev_ctx.template Alloc<T>(input_grad);
  if (filter_grad) dev_ctx.template Alloc<T>(filter_grad);
  if (bias_grad) dev_ctx.template Alloc<T>(bias_grad);
  const bool channel_last = (data_format == "NHWC");

  std::vector<int> strides = strides_t;
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;

  // Update Padding And Dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();

  DDim in_data_dims;
  const DataLayout data_layout = common::StringToDataLayout(data_format);
  if (data_layout != DataLayout::NHWC) {
    in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  } else {
    in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
  }
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  //  [Top, Bottom, Left, Right] -> [Top, Left]
  bool is_sys_pad = strides.size() * 2 == paddings.size() ? false : true;
  if (!is_sys_pad) {
    for (size_t i = 0; i < strides.size(); ++i) {
      paddings.erase(paddings.begin() + i + 1);
    }
  }

  LaunchDepthwiseConv2dBackwardCompatible<T, Context>(dev_ctx,
                                                      input,
                                                      filter,
                                                      out_grad,
                                                      strides,
                                                      paddings,
                                                      dilations,
                                                      data_format,
                                                      input_grad,
                                                      filter_grad,
                                                      bias_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(depthwise_conv2d_bias_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv2dBiasGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
