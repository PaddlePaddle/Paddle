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
#include <limits>
#include "paddle/common/enforce.h"
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

template <typename Context>
inline uint32_t GET_BLOCKS(
    const Context& dev_ctx,
    const int64_t N,
    const int64_t max_threads_per_block = CUDA_NUM_THREADS) {
  const int64_t block_num = (N - 1) / max_threads_per_block + 1;
  PADDLE_ENFORCE_LE_UINT32_MAX(block_num, "block_num");
  PADDLE_ENFORCE_LE(
      block_num,
      static_cast<int64_t>(dev_ctx.GetCUDAMaxGridDimSize()[0]),
      common::errors::InvalidArgument(
          "depthwise conv2d bias grad grid.x exceeds device limit."));
  return static_cast<uint32_t>(block_num);
}

inline int GetGradParamsNumThreads(int64_t batchSize) {
  constexpr int MAX_BLOCK_SIZE = 256;
  return static_cast<int>(
      std::min(static_cast<int64_t>(batchSize) * CUDA_WARP_SIZE,
               static_cast<int64_t>(MAX_BLOCK_SIZE)));
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
__device__ __forceinline__ dtype::float16 WARP_SHFL_DOWN<dtype::float16>(
    dtype::float16 value, unsigned int delta, int width, unsigned int mask) {
  uint16_t val_as_ushort = *reinterpret_cast<uint16_t*>(&value);
  uint16_t shuffled =
      WARP_SHFL_DOWN<uint16_t>(val_as_ushort, delta, width, mask);
  return *reinterpret_cast<dtype::float16*>(&shuffled);
}

template <>
__device__ __forceinline__ dtype::bfloat16 WARP_SHFL_DOWN<dtype::bfloat16>(
    dtype::bfloat16 value, unsigned int delta, int width, unsigned int mask) {
  uint16_t val_as_ushort = *reinterpret_cast<uint16_t*>(&value);
  uint16_t shuffled =
      WARP_SHFL_DOWN<uint16_t>(val_as_ushort, delta, width, mask);
  return *reinterpret_cast<dtype::bfloat16*>(&shuffled);
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
  using AccT = typename MPTypeTrait<T>::Type;
  const int KW_LIMIT = (kSize != 0) ? kSize : kernelWidth;
  const int KH_LIMIT = (kSize != 0) ? kSize : kernelHeight;
  const int strideW = (stride != 0) ? stride : strideWidth;
  const int strideH = (stride != 0) ? stride : strideHeight;

  for (IndexT linearIndex =
           static_cast<IndexT>(blockIdx.x) * static_cast<IndexT>(blockDim.x) +
           static_cast<IndexT>(threadIdx.x);
       linearIndex < totalElements;
       linearIndex +=
       static_cast<IndexT>(blockDim.x) * static_cast<IndexT>(gridDim.x)) {
    IndexT indtmp1 = linearIndex / inputWidth;
    const int w = static_cast<int>(linearIndex - indtmp1 * inputWidth);
    IndexT indtmp2 = indtmp1 / inputHeight;
    const int h = static_cast<int>(indtmp1 - indtmp2 * inputHeight);
    indtmp1 = indtmp2;
    indtmp2 = indtmp1 / inputChannels;
    const int c = static_cast<int>(indtmp1 - indtmp2 * inputChannels);
    const IndexT n = indtmp2;

    AccT value(0);

    for (int multiplier = 0; multiplier < depthwiseMultiplier; ++multiplier) {
      const int och = c * depthwiseMultiplier + multiplier;
      IndexT weightOffset =
          static_cast<IndexT>(och) * kernelHeight * kernelWidth;
      for (int kh = 0; kh < KH_LIMIT; ++kh) {
        for (int kw = 0; kw < KW_LIMIT; ++kw) {
          int64_t h_out = static_cast<int64_t>(h) + padHeight -
                          static_cast<int64_t>(kh) * dilationHeight;
          int64_t w_out = static_cast<int64_t>(w) + padWidth -
                          static_cast<int64_t>(kw) * dilationWidth;

          if ((h_out % strideH == 0) && (w_out % strideW == 0)) {
            h_out = h_out / strideH;
            w_out = w_out / strideW;

            if ((h_out >= 0) && (h_out < outputHeight) && (w_out >= 0) &&
                (w_out < outputWidth)) {
              const IndexT offset = ((n * outputChannels + och) * outputHeight +
                                     static_cast<IndexT>(h_out)) *
                                        outputWidth +
                                    static_cast<IndexT>(w_out);
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
  using AccT = typename MPTypeTrait<T>::Type;
  const int64_t channelStride =
      static_cast<int64_t>(kernelWidth) * kernelHeight;

  const int64_t bidx = blockIdx.x;
  int kW = static_cast<int>(bidx % kernelWidth);
  int kH = static_cast<int>((bidx / kernelWidth) % kernelHeight);
  int ch = static_cast<int>(bidx / channelStride);

  int inputCh = ch / depthwiseMultiplier;

  AccT grad(0);

  const int laneId = threadIdx.x % CUDA_WARP_SIZE;
  const int batch = threadIdx.x / CUDA_WARP_SIZE;
  const int nwarps = blockDim.x / CUDA_WARP_SIZE;
  const int64_t imageElements =
      static_cast<int64_t>(outputWidth) * outputHeight;

  for (int batchIdx = batch; batchIdx < batchSize; batchIdx += nwarps) {
    for (int64_t idx = laneId; idx < imageElements; idx += CUDA_WARP_SIZE) {
      const int64_t go_w_offset = idx % outputWidth;
      const int64_t go_h_offset = idx / outputWidth;

      const int64_t i_w_offset = go_w_offset * strideWidth +
                                 static_cast<int64_t>(kW) * dilationWidth -
                                 padWidth;
      const int64_t i_h_offset = go_h_offset * strideHeight +
                                 static_cast<int64_t>(kH) * dilationHeight -
                                 padHeight;

      if (i_w_offset >= 0 && i_h_offset >= 0 && i_w_offset < inputWidth &&
          i_h_offset < inputHeight) {
        const int64_t inputOffset =
            ((static_cast<int64_t>(batchIdx) * inputChannels + inputCh) *
                 inputHeight +
             i_h_offset) *
                inputWidth +
            i_w_offset;
        const int64_t outputOffset =
            ((static_cast<int64_t>(batchIdx) * kernelChannels + ch) *
             outputHeight) *
                outputWidth +
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
    const int64_t weightOffset =
        kW + static_cast<int64_t>(kernelWidth) * kH +
        static_cast<int64_t>(kernelWidth) * kernelHeight * ch;
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

  PADDLE_ENFORCE_LE_INT_MAX(batchSize, "batchSize");
  PADDLE_ENFORCE_LE_INT_MAX(c_in, "c_in");
  PADDLE_ENFORCE_LE_INT_MAX(h_in, "h_in");
  PADDLE_ENFORCE_LE_INT_MAX(w_in, "w_in");
  PADDLE_ENFORCE_LE_INT_MAX(outputChannels, "outputChannels");
  PADDLE_ENFORCE_LE_INT_MAX(h_out, "h_out");
  PADDLE_ENFORCE_LE_INT_MAX(w_out, "w_out");
  PADDLE_ENFORCE_LE_INT_MAX(kH, "kH");
  PADDLE_ENFORCE_LE_INT_MAX(kW, "kW");
  PADDLE_ENFORCE_LE_INT_MAX(depthwiseMultiplier, "depthwiseMultiplier");

  const int batchSize_int = static_cast<int>(batchSize);
  const int c_in_int = static_cast<int>(c_in);
  const int h_in_int = static_cast<int>(h_in);
  const int w_in_int = static_cast<int>(w_in);
  const int outputChannels_int = static_cast<int>(outputChannels);
  const int h_out_int = static_cast<int>(h_out);
  const int w_out_int = static_cast<int>(w_out);
  const int kH_int = static_cast<int>(kH);
  const int kW_int = static_cast<int>(kW);
  const int depthwiseMultiplier_int = static_cast<int>(depthwiseMultiplier);

  // Launch Filter Gradient Kernel (grad_weight)
  if (filter_grad_nchw_ptr) {
    funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, filter_grad_nchw_ptr, static_cast<T>(0));

    const int64_t blocks = outputChannels * kH * kW;
    PADDLE_ENFORCE_LE_UINT32_MAX(blocks, "depthwise conv2d bias grad grid.x");
    PADDLE_ENFORCE_LE(
        blocks,
        static_cast<int64_t>(dev_ctx.GetCUDAMaxGridDimSize()[0]),
        common::errors::InvalidArgument(
            "depthwise conv2d bias grad grid.x exceeds device limit."));
    dim3 grid(static_cast<uint32_t>(blocks));
    const int threads = GetGradParamsNumThreads(batchSize);
    dim3 block(threads);

    size_t smem =
        (block.x / CUDA_WARP_SIZE) * sizeof(typename MPTypeTrait<T>::Type);

    DWConv2dBwdWeightKernel<T, int64_t>
        <<<grid, block, smem, stream>>>(out_grad_nchw.data<T>(),
                                        input_nchw.data<T>(),
                                        filter_grad_nchw_ptr->data<T>(),
                                        batchSize_int,
                                        c_in_int,
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

  // Launch Input Gradient Kernel (grad_input)
  if (input_grad_nchw_ptr) {
    int64_t totalElements = input_grad_nchw_ptr->numel();

    const T* grad_output_ptr = out_grad_nchw.data<T>();
    T* grad_input_ptr = input_grad_nchw_ptr->data<T>();
    const T* weight_ptr = filter_nchw.data<T>();

    uint32_t blocks = GET_BLOCKS(dev_ctx, totalElements, CUDA_NUM_THREADS);
    const int64_t input_grad_step =
        static_cast<int64_t>(blocks) * CUDA_NUM_THREADS;
    const bool use_int32_input_kernel =
        totalElements <= std::numeric_limits<int>::max() &&
        out_grad_nchw.numel() <= std::numeric_limits<int>::max() &&
        filter_nchw.numel() <= std::numeric_limits<int>::max() &&
        input_grad_step <= std::numeric_limits<int>::max();

#define LAUNCH_INPUT_KERNEL(K, S)                                             \
  if (use_int32_input_kernel) {                                               \
    DWConv2dBwdInputKernel<K, S, T, int>                                      \
        <<<dim3(blocks), dim3(CUDA_NUM_THREADS), 0, stream>>>(                \
            grad_output_ptr,                                                  \
            grad_input_ptr,                                                   \
            weight_ptr,                                                       \
            static_cast<int>(totalElements),                                  \
            c_in_int,                                                         \
            depthwiseMultiplier_int,                                          \
            outputChannels_int,                                               \
            w_in_int,                                                         \
            h_in_int,                                                         \
            w_out_int,                                                        \
            h_out_int,                                                        \
            kW_int,                                                           \
            kH_int,                                                           \
            strideW,                                                          \
            strideH,                                                          \
            padW,                                                             \
            padH,                                                             \
            dW,                                                               \
            dH);                                                              \
  } else {                                                                    \
    constexpr int kInputGradInt64NumThreads = 512;                            \
    const uint32_t blocks_int64 =                                             \
        GET_BLOCKS(dev_ctx, totalElements, kInputGradInt64NumThreads);        \
    DWConv2dBwdInputKernel<K, S, T, int64_t>                                  \
        <<<dim3(blocks_int64), dim3(kInputGradInt64NumThreads), 0, stream>>>( \
            grad_output_ptr,                                                  \
            grad_input_ptr,                                                   \
            weight_ptr,                                                       \
            totalElements,                                                    \
            c_in_int,                                                         \
            depthwiseMultiplier_int,                                          \
            outputChannels_int,                                               \
            w_in_int,                                                         \
            h_in_int,                                                         \
            w_out_int,                                                        \
            h_out_int,                                                        \
            kW_int,                                                           \
            kH_int,                                                           \
            strideW,                                                          \
            strideH,                                                          \
            padW,                                                             \
            padH,                                                             \
            dW,                                                               \
            dH);                                                              \
  }

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

    SumKernel<T, Context>(dev_ctx,
                          out_grad_nchw,
                          IntArray(reduce_dims),
                          CppTypeToDataType<T>::Type(),
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
                                   const optional<DenseTensor>& bias,
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
