// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/correlation_funcs.cu.h"

namespace phi {

template <typename T>
__global__ void correlation_forward(T *output,
                                    const int output_channel,
                                    const int output_height,
                                    const int output_width,
                                    const T *rinput1,
                                    const int input_channel,
                                    const int input_height,
                                    const int input_width,
                                    const T *rinput2,
                                    const int pad_size,
                                    const int kernel_size,
                                    const int max_displacement,
                                    const int stride1,
                                    const int stride2) {
  int p_input_width = input_width + 2 * pad_size;
  int p_input_height = input_height + 2 * pad_size;

  int kernel_rad = (kernel_size - 1) / 2;
  int displacement_rad = max_displacement / stride2;

  int displacement_size = 2 * displacement_rad + 1;

  int n = blockIdx.x;
  int h1 = blockIdx.y * stride1 + max_displacement;
  int w1 = blockIdx.z * stride1 + max_displacement;
  int c = threadIdx.x;

  int p_dimchw = p_input_height * p_input_width * input_channel;
  int p_dimcw = p_input_width * input_channel;
  int p_dimc = input_channel;

  int t_dimchw = output_channel * output_height * output_width;
  int t_dimhw = output_height * output_width;
  int t_dimw = output_width;

  int nelems = kernel_size * kernel_size * p_dimc;

  for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
    for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
      int w2 = w1 + ti * stride2;
      int h2 = h1 + tj * stride2;

      T acc0 = 0;
      for (int j = -kernel_rad; j <= kernel_rad; ++j) {
        for (int i = -kernel_rad; i <= kernel_rad; ++i) {
          for (int ch = c; ch < p_dimc; ch += blockDim.x) {
            int index1 =
                n * p_dimchw + (h1 + j) * p_dimcw + (w1 + i) * p_dimc + ch;
            int index2 =
                n * p_dimchw + (h2 + j) * p_dimcw + (w2 + i) * p_dimc + ch;
            acc0 += static_cast<T>(rinput1[index1] * rinput2[index2]);
          }
        }
      }
      if (blockDim.x == warpSize) {
        __syncwarp();
        acc0 = warpReduceSum(acc0);
      } else {
        __syncthreads();
        acc0 = blockReduceSum(acc0);
      }

      if (threadIdx.x == 0) {
        int tc = (tj + displacement_rad) * displacement_size +
                 (ti + displacement_rad);
        const int t_index =
            n * t_dimchw + tc * t_dimhw + blockIdx.y * t_dimw + blockIdx.z;
        output[t_index] = static_cast<T>(acc0 / nelems);
      }
    }
  }
}

template <typename T, typename Context>
void CorrelationCUDAKernel(const Context &dev_ctx,
                           const DenseTensor &input1,
                           const DenseTensor &input2,
                           int pad_size,
                           int kernel_size,
                           int max_displacement,
                           int stride1,
                           int stride2,
                           int corr_type_multiply,
                           DenseTensor *out) {
  bool is_gpu_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU;
  PADDLE_ENFORCE_EQ(
      is_gpu_place,
      true,
      common::errors::InvalidArgument("Correlation only supports GPU now."));

  dev_ctx.template Alloc<T>(out);

  // base on input1, NCHW
  auto in_dims = input1.dims();
  int N = in_dims[0];
  int C = in_dims[1];
  int H = in_dims[2];
  int W = in_dims[3];

  int padded_input_height = H + 2 * pad_size;
  int padded_input_width = W + 2 * pad_size;

  phi::DenseTensor rinput1;
  rinput1.Resize({N, padded_input_height, padded_input_width, C});
  dev_ctx.template Alloc<T>(&rinput1);

  phi::DenseTensor rinput2;
  rinput2.Resize({N, padded_input_height, padded_input_width, C});
  dev_ctx.template Alloc<T>(&rinput2);

  set_zero<<<(rinput1.numel() + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(
      rinput1.data<T>(), rinput1.numel());
  set_zero<<<(rinput2.numel() + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(
      rinput2.data<T>(), rinput2.numel());
  set_zero<<<(out->numel() + 512 - 1) / 512, 512, 0, dev_ctx.stream()>>>(
      out->data<T>(), out->numel());

  auto out_dims = out->dims();
  int OC = out_dims[1];
  int OH = out_dims[2];
  int OW = out_dims[3];

  dim3 blocks_grid(N, H, W);
  dim3 threads_block(THREADS_PER_BLOCK);

  channel_first<T><<<blocks_grid, threads_block, 0, dev_ctx.stream()>>>(
      input1.data<T>(), rinput1.data<T>(), C, H, W, pad_size);
  channel_first<T><<<blocks_grid, threads_block, 0, dev_ctx.stream()>>>(
      input2.data<T>(), rinput2.data<T>(), C, H, W, pad_size);

  dim3 threadsPerBlock(THREADS_PER_BLOCK);
  dim3 totalBlocksCorr(N, OH, OW);

  correlation_forward<T>
      <<<totalBlocksCorr, threadsPerBlock, 0, dev_ctx.stream()>>>(
          out->data<T>(),
          OC,
          OH,
          OW,
          rinput1.data<T>(),
          C,
          H,
          W,
          rinput2.data<T>(),
          pad_size,
          kernel_size,
          max_displacement,
          stride1,
          stride2);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    correlation, GPU, ALL_LAYOUT, phi::CorrelationCUDAKernel, float, double) {}
