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

template <typename T,
          typename AccT,
          int kKnownKernelT,
          int kKnownKernelH,
          int kKnownKernelW,
          int kKnownDilationT,
          int kKnownDilationH,
          int kKnownDilationW,
          int kKnownStrideT,
          int kKnownStrideH,
          int kKnownStrideW>
__global__ void DWConv3dBwdInputKernel(const T* grad_output,
                                       T* grad_input,
                                       const T* weight,
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
                                       const int stride_t_in,
                                       int stride_h_in,
                                       int stride_w_in,
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
  const int stride_t = kKnownStrideT > 0 ? kKnownStrideT : stride_t_in;
  const int stride_h = kKnownStrideH > 0 ? kKnownStrideH : stride_h_in;
  const int stride_w = kKnownStrideW > 0 ? kKnownStrideW : stride_w_in;

  const int channel_multiplier = output_channels / input_channels;
  const int num_input =
      batch_size * input_channels * input_depth * input_height * input_width;

  const int i_stride_c = input_depth * input_height * input_width;
  const int i_stride_d = input_height * input_width;
  const int i_stride_h = input_width;

  const int o_stride_c = output_depth * output_height * output_width;
  const int o_stride_d = output_height * output_width;
  const int o_stride_h = output_width;

  const int w_stride_c = kernel_t * kernel_h * kernel_w;

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_input;
       index += blockDim.x * gridDim.x) {
    int temp = index;
    const int in_col = temp % input_width;
    temp /= input_width;
    const int in_row = temp % input_height;
    temp /= input_height;
    const int in_frame = temp % input_depth;
    temp /= input_depth;
    const int in_channel = temp % input_channels;
    const int batch = temp / input_channels;

    const int out_col_end = in_col + padding_w;
    const int out_row_end = in_row + padding_h;
    const int out_frame_end = in_frame + padding_t;

    AccT sum = 0;
    const T* weight_ptr =
        weight + (in_channel * channel_multiplier) * w_stride_c;
    const T* gout_base = grad_output + batch * (output_channels * o_stride_c);

    for (int k_chn = 0; k_chn < channel_multiplier; ++k_chn) {
      const T* gout_ptr =
          gout_base + (in_channel * channel_multiplier + k_chn) * o_stride_c;

      for (int k_frame = 0; k_frame < kernel_t; ++k_frame) {
        const int out_frame_raw = out_frame_end - k_frame * dilation_t;
        const int out_frame = out_frame_raw / stride_t;

        if (out_frame * stride_t == out_frame_raw && out_frame >= 0 &&
            out_frame < output_depth) {
          for (int k_row = 0; k_row < kernel_h; ++k_row) {
            const int out_row_raw = out_row_end - k_row * dilation_h;
            const int out_row = out_row_raw / stride_h;

            if (out_row * stride_h == out_row_raw && out_row >= 0 &&
                out_row < output_height) {
              for (int k_col = 0; k_col < kernel_w; ++k_col) {
                const int out_col_raw = out_col_end - k_col * dilation_w;
                const int out_col = out_col_raw / stride_w;

                if (out_col * stride_w == out_col_raw && out_col >= 0 &&
                    out_col < output_width) {
                  int w_offset = k_chn * w_stride_c +
                                 k_frame * kernel_h * kernel_w +
                                 k_row * kernel_w + k_col;
                  T val_w = weight_ptr[w_offset];

                  int out_offset =
                      out_frame * o_stride_d + out_row * o_stride_h + out_col;
                  T val_go = gout_ptr[out_offset];

                  sum += static_cast<AccT>(val_w) * static_cast<AccT>(val_go);
                }
              }
            }
          }
        }
      }
    }
    grad_input[index] = static_cast<T>(sum);
  }
}

template <typename T, typename AccT, int kKnownStrideH, int kKnownStrideW>
__global__ void DWConv3dBwdWeightKernel(const T* grad_output,
                                        const T* input,
                                        T* grad_weight,
                                        const int batch_size,
                                        const int input_channels,
                                        const int input_depth,
                                        int input_height,
                                        int input_width,
                                        const int output_channels,
                                        const int output_depth,
                                        int output_height,
                                        int output_width,
                                        const int kernel_t,
                                        int kernel_h,
                                        int kernel_w,
                                        const int stride_t,
                                        int stride_h_in,
                                        int stride_w_in,
                                        const int padding_t,
                                        int padding_h,
                                        int padding_w,
                                        const int dilation_t,
                                        int dilation_h,
                                        int dilation_w) {
  const int stride_h = kKnownStrideH > 0 ? kKnownStrideH : stride_h_in;
  const int stride_w = kKnownStrideW > 0 ? kKnownStrideW : stride_w_in;

  const int k_c = output_channels;
  const int k_col = blockIdx.x % kernel_w;
  const int k_row = (blockIdx.x / kernel_w) % kernel_h;
  const int k_frame = (blockIdx.x / kernel_w / kernel_h) % kernel_t;
  const int k_channel = blockIdx.x / kernel_w / kernel_h / kernel_t;

  if (k_channel >= k_c) return;

  const int channel_multiplier = output_channels / input_channels;
  const int in_channel = k_channel / channel_multiplier;

  extern __shared__ char smem_raw[];
  T* sdata = reinterpret_cast<T*>(smem_raw);

  const int laneid = threadIdx.x % CUDA_WARP_SIZE;
  const int warpid = threadIdx.x / CUDA_WARP_SIZE;
  const int nwarps = blockDim.x / CUDA_WARP_SIZE;

  const int i_stride_c = input_depth * input_height * input_width;
  const int i_stride_d = input_height * input_width;
  const int i_stride_h = input_width;

  const int o_stride_c = output_depth * output_height * output_width;
  const int o_stride_d = output_height * output_width;
  const int o_stride_h = output_width;

  AccT grad = 0;

  int batch = warpid / output_depth;
  int gout_frame = warpid - batch * output_depth;

  const int total_outer_loops = batch_size * output_depth;

  for (int outer_pos = warpid; outer_pos < total_outer_loops;
       outer_pos += nwarps, gout_frame += nwarps) {
    while (gout_frame >= output_depth) {
      gout_frame -= output_depth;
      batch++;
    }

    const int in_frame =
        (gout_frame * stride_t) + (k_frame * dilation_t) - padding_t;

    if (in_frame >= 0 && in_frame < input_depth) {
      const T* gout_ptr = grad_output + batch * (output_channels * o_stride_c) +
                          k_channel * o_stride_c + gout_frame * o_stride_d;

      const T* input_ptr = input + batch * (input_channels * i_stride_c) +
                           in_channel * i_stride_c + in_frame * i_stride_d;

      const T* gout_ptr_lane = gout_ptr + laneid;

      int gout_row = laneid / output_width;
      int gout_col = laneid % output_width;

      for (; gout_row < output_height;) {
        const AccT op1 = static_cast<AccT>(*gout_ptr_lane);
        gout_ptr_lane += CUDA_WARP_SIZE;

        const int in_col =
            (gout_col * stride_w) + (k_col * dilation_w) - padding_w;
        const int in_row =
            (gout_row * stride_h) + (k_row * dilation_h) - padding_h;

        AccT op2 = 0;
        if (in_row >= 0 && in_row < input_height && in_col >= 0 &&
            in_col < input_width) {
          const int in_pos = in_row * i_stride_h + in_col;
          op2 = static_cast<AccT>(input_ptr[in_pos]);
        }

        grad += op1 * op2;

        gout_col += CUDA_WARP_SIZE;
        while (gout_col >= output_width) {
          gout_col -= output_width;
          gout_row++;
        }
      }
    }
  }

  sdata[threadIdx.x] = static_cast<T>(grad);
  __syncthreads();

  for (int i = blockDim.x / 2; i >= 1; i >>= 1) {
    if (threadIdx.x < i) {
      AccT val = static_cast<AccT>(sdata[threadIdx.x]) +
                 static_cast<AccT>(sdata[threadIdx.x + i]);
      sdata[threadIdx.x] = static_cast<T>(val);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    grad_weight[blockIdx.x] = sdata[0];
  }
}

template <typename T, typename Context>
void LaunchDepthwiseConv3dBackwardCompatible(const Context& dev_ctx,
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
  const bool channel_last = (data_format == "NDHWC");

  DenseTensor input_ncdhw;
  DenseTensor out_grad_ncdhw;
  const DenseTensor& filter_ncdhw = filter;

  DenseTensor input_grad_ncdhw_tmp;
  DenseTensor* input_grad_ncdhw_ptr = nullptr;
  DenseTensor* filter_grad_ncdhw_ptr = nullptr;

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(dev_ctx, &input, &input_ncdhw);
    TransToChannelFirst<Context, T>(dev_ctx, &input, &input_ncdhw);

    ResizeToChannelFirst<Context, T>(dev_ctx, &out_grad, &out_grad_ncdhw);
    TransToChannelFirst<Context, T>(dev_ctx, &out_grad, &out_grad_ncdhw);

    if (input_grad) {
      ResizeToChannelFirst<Context, T>(
          dev_ctx, input_grad, &input_grad_ncdhw_tmp);
      dev_ctx.template Alloc<T>(&input_grad_ncdhw_tmp);
      input_grad_ncdhw_ptr = &input_grad_ncdhw_tmp;
    }
  } else {
    input_ncdhw.ShareDataWith(input);
    out_grad_ncdhw.ShareDataWith(out_grad);
    if (input_grad) input_grad_ncdhw_ptr = input_grad;
  }

  if (filter_grad) {
    if (channel_last) dev_ctx.template Alloc<T>(filter_grad);
    filter_grad_ncdhw_ptr = filter_grad;
  }

  const int64_t batch_size = input_ncdhw.dims()[0];
  const int64_t in_channels = input_ncdhw.dims()[1];
  const int64_t in_depth = input_ncdhw.dims()[2];
  const int64_t in_height = input_ncdhw.dims()[3];
  const int64_t in_width = input_ncdhw.dims()[4];

  const int64_t out_channels = out_grad_ncdhw.dims()[1];
  const int64_t out_depth = out_grad_ncdhw.dims()[2];
  const int64_t out_height = out_grad_ncdhw.dims()[3];
  const int64_t out_width = out_grad_ncdhw.dims()[4];

  const int64_t kernel_t = filter_ncdhw.dims()[2];
  const int64_t kernel_h = filter_ncdhw.dims()[3];
  const int64_t kernel_w = filter_ncdhw.dims()[4];
  std::vector<int> kernel_size = {static_cast<int>(kernel_t),
                                  static_cast<int>(kernel_h),
                                  static_cast<int>(kernel_w)};

  std::vector<int> paddings_vec;
  if (paddings.size() == 6) {
    paddings_vec = {paddings[0], paddings[2], paddings[4]};
  } else if (paddings.size() == 3) {
    paddings_vec = paddings;
  } else {
    paddings_vec = {paddings[0], paddings[0], paddings[0]};
  }

  auto stream = dev_ctx.stream();
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;

  const T* input_ptr = input_ncdhw.data<T>();
  const T* grad_output_ptr = out_grad_ncdhw.data<T>();
  const T* filter_ptr = filter_ncdhw.data<T>();

  T* grad_input_ptr =
      input_grad_ncdhw_ptr ? input_grad_ncdhw_ptr->data<T>() : nullptr;
  T* grad_filter_ptr =
      filter_grad_ncdhw_ptr ? filter_grad_ncdhw_ptr->data<T>() : nullptr;

  // Input Gradient
  if (grad_input_ptr) {
    int64_t num_elements = input_grad_ncdhw_ptr->numel();
    int block = 256;
    int grid = std::min((num_elements - 1) / block + 1, (int64_t)65536);

    bool is_k3 = (kernel_t == 3 && kernel_h == 3 && kernel_w == 3);
    bool is_d1 = (dilations[0] == 1 && dilations[1] == 1 && dilations[2] == 1);
    bool is_s1 = (strides[0] == 1 && strides[1] == 1 && strides[2] == 1);

    if (is_k3 && is_d1 && is_s1) {
      DWConv3dBwdInputKernel<T, AccT, 3, 3, 3, 1, 1, 1, 1, 1, 1>
          <<<grid, block, 0, stream>>>(grad_output_ptr,
                                       grad_input_ptr,
                                       filter_ptr,
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
    } else if (is_k3 && is_d1) {
      DWConv3dBwdInputKernel<T, AccT, 3, 3, 3, 1, 1, 1, -1, -1, -1>
          <<<grid, block, 0, stream>>>(grad_output_ptr,
                                       grad_input_ptr,
                                       filter_ptr,
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
    } else if (is_k3 && is_s1) {
      DWConv3dBwdInputKernel<T, AccT, 3, 3, 3, -1, -1, -1, 1, 1, 1>
          <<<grid, block, 0, stream>>>(grad_output_ptr,
                                       grad_input_ptr,
                                       filter_ptr,
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
    } else if (is_k3) {
      DWConv3dBwdInputKernel<T, AccT, 3, 3, 3, -1, -1, -1, -1, -1, -1>
          <<<grid, block, 0, stream>>>(grad_output_ptr,
                                       grad_input_ptr,
                                       filter_ptr,
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
      DWConv3dBwdInputKernel<T, AccT, -1, -1, -1, -1, -1, -1, -1, -1, -1>
          <<<grid, block, 0, stream>>>(grad_output_ptr,
                                       grad_input_ptr,
                                       filter_ptr,
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
  }

  // Weight Gradient
  if (grad_filter_ptr) {
    int64_t num_weights = filter_grad_ncdhw_ptr->numel();
    int block = 256;
    int grid = num_weights;
    size_t smem = block * sizeof(T);

    if (strides[1] == 1 && strides[2] == 1) {
      DWConv3dBwdWeightKernel<T, AccT, 1, 1>
          <<<grid, block, smem, stream>>>(grad_output_ptr,
                                          input_ptr,
                                          grad_filter_ptr,
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
    } else if (strides[1] == 2 && strides[2] == 2) {
      DWConv3dBwdWeightKernel<T, AccT, 2, 2>
          <<<grid, block, smem, stream>>>(grad_output_ptr,
                                          input_ptr,
                                          grad_filter_ptr,
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
      DWConv3dBwdWeightKernel<T, AccT, -1, -1>
          <<<grid, block, smem, stream>>>(grad_output_ptr,
                                          input_ptr,
                                          grad_filter_ptr,
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
  }

  // Bias Gradient
  if (bias_grad) {
    dev_ctx.template Alloc<T>(bias_grad);
    // Reduce N(0), D(2), H(3), W(4) -> C(1) for NCDHW
    std::vector<int64_t> reduce_dims = {0, 2, 3, 4};
    phi::SumKernel<T, Context>(dev_ctx,
                               out_grad_ncdhw,
                               phi::IntArray(reduce_dims),
                               CppTypeToDataType<T>::Type(),
                               false,
                               bias_grad);
  }

  if (input_grad && channel_last) {
    TransToChannelLast<Context, T>(dev_ctx, input_grad_ncdhw_ptr, input_grad);
  }
}

template <typename T, typename Context>
void DepthwiseConv3dBiasGradKernel(const Context& dev_ctx,
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
  if (!input_grad && !filter_grad && !bias_grad) return;

  // 0-size Check
  if (input.numel() == 0) {
    if (input_grad) dev_ctx.template Alloc<T>(input_grad);
    if (filter_grad) {
      dev_ctx.template Alloc<T>(filter_grad);
      Full<T, Context>(dev_ctx, filter_grad->dims(), 0, filter_grad);
    }
    if (bias_grad) {
      dev_ctx.template Alloc<T>(bias_grad);
      Full<T, Context>(dev_ctx, bias_grad->dims(), 0, bias_grad);
    }
    return;
  }

  // Alloc
  if (input_grad) dev_ctx.template Alloc<T>(input_grad);
  if (filter_grad) dev_ctx.template Alloc<T>(filter_grad);

  std::vector<int> strides = strides_t;
  std::vector<int> dilations = dilations_t;
  std::vector<int> paddings = paddings_t;

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

  bool is_sys_pad = strides.size() * 2 == paddings.size() ? false : true;
  if (!is_sys_pad) {
    for (size_t i = 0; i < strides.size(); ++i) {
      paddings.erase(paddings.begin() + i + 1);
    }
  }

  LaunchDepthwiseConv3dBackwardCompatible<T, Context>(dev_ctx,
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

PD_REGISTER_KERNEL(depthwise_conv3d_bias_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv3dBiasGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
