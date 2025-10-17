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

#include "paddle/phi/kernels/gpu/correlation_kernel.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/correlation_funcs.cu.h"
namespace phi {

template <typename T>
__global__ void correlation_forward(T *output,
                                    const int64_t output_channel,
                                    const int64_t output_height,
                                    const int64_t output_width,
                                    const T *rinput1,
                                    const int64_t input_channel,
                                    const int64_t input_height,
                                    const int64_t input_width,
                                    const T *rinput2,
                                    const int pad_size,
                                    const int kernel_size,
                                    const int max_displacement,
                                    const int stride1,
                                    const int stride2,
                                    const int OH,
                                    const int OW) {
  int64_t p_input_width = input_width + 2 * pad_size;
  int64_t p_input_height = input_height + 2 * pad_size;

  int kernel_rad = (kernel_size - 1) / 2;
  int displacement_rad = max_displacement / stride2;
  int displacement_size = 2 * displacement_rad + 1;

  int64_t global_block_id = blockIdx.x;
  int64_t hw = (int64_t)OH * OW;

  int64_t n = global_block_id / hw;
  int64_t hw_index = global_block_id % hw;

  int64_t h1 = (hw_index / OW) * stride1 + max_displacement;
  int64_t w1 = (hw_index % OW) * stride1 + max_displacement;

  int64_t c = threadIdx.x;

  int64_t p_dimchw = p_input_height * p_input_width * input_channel;
  int64_t p_dimcw = p_input_width * input_channel;
  int64_t p_dimc = input_channel;

  int64_t t_dimchw = output_channel * output_height * output_width;
  int64_t t_dimhw = output_height * output_width;
  int64_t t_dimw = output_width;

  int64_t nelems = kernel_size * kernel_size * p_dimc;

  for (int64_t tj = -displacement_rad; tj <= displacement_rad; ++tj) {
    for (int64_t ti = -displacement_rad; ti <= displacement_rad; ++ti) {
      int64_t w2 = w1 + ti * stride2;
      int64_t h2 = h1 + tj * stride2;

      T acc0 = 0;
      for (int j = -kernel_rad; j <= kernel_rad; ++j) {
        for (int i = -kernel_rad; i <= kernel_rad; ++i) {
          for (int ch = c; ch < p_dimc; ch += blockDim.x) {
            int64_t index1 =
                n * p_dimchw + (h1 + j) * p_dimcw + (w1 + i) * p_dimc + ch;
            int64_t index2 =
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
        int64_t tc = (tj + displacement_rad) * displacement_size +
                     (ti + displacement_rad);
        const int64_t t_index = n * t_dimchw + tc * t_dimhw +
                                (h1 - max_displacement) / stride1 * t_dimw +
                                (w1 - max_displacement) / stride1;
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
  bool is_gpu_place =
      dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU ||
      dev_ctx.GetPlace().GetType() == phi::AllocationType::CUSTOM;
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

  auto gplace = phi::GPUPlace(phi::backends::gpu::GetCurrentDeviceId());
  auto *ctx =
      static_cast<GPUContext *>(phi::DeviceContextPool::Instance().Get(gplace));
  auto max_grid_dim = static_cast<int64_t>(dev_ctx.GetCUDAMaxGridDimSize()[0]);

  int64_t grid_size = (rinput1.numel() + 512 - 1) / 512;
  grid_size = std::min(static_cast<int64_t>(grid_size), max_grid_dim);
  set_zero<<<grid_size, 512, 0, dev_ctx.stream()>>>(rinput1.data<T>(),
                                                    rinput1.numel());

  grid_size = std::min(static_cast<int64_t>((rinput2.numel() + 512 - 1) / 512),
                       max_grid_dim);
  set_zero<<<grid_size, 512, 0, dev_ctx.stream()>>>(rinput2.data<T>(),
                                                    rinput2.numel());

  grid_size = std::min(static_cast<int64_t>((out->numel() + 512 - 1) / 512),
                       max_grid_dim);
  set_zero<<<grid_size, 512, 0, dev_ctx.stream()>>>(out->data<T>(),
                                                    out->numel());

  auto out_dims = out->dims();
  int OC = out_dims[1];
  int OH = out_dims[2];
  int OW = out_dims[3];

  int blocks_grid = std::min(static_cast<int64_t>(N) * H * W, max_grid_dim);
  dim3 threads_block(THREADS_PER_BLOCK);

  channel_first<T><<<blocks_grid, threads_block, 0, dev_ctx.stream()>>>(
      input1.data<T>(), rinput1.data<T>(), N, C, H, W, pad_size);
  channel_first<T><<<blocks_grid, threads_block, 0, dev_ctx.stream()>>>(
      input2.data<T>(), rinput2.data<T>(), N, C, H, W, pad_size);

  dim3 threadsPerBlock(THREADS_PER_BLOCK);
  // dim3 totalBlocksCorr(N, OH, OW);
  grid_size = std::min(static_cast<int64_t>(N) * OH * OW, max_grid_dim);

  correlation_forward<T>
      <<<grid_size, threadsPerBlock, 0, dev_ctx.stream()>>>(out->data<T>(),
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
                                                            stride2,
                                                            OH,
                                                            OW);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    correlation, GPU, ALL_LAYOUT, phi::CorrelationCUDAKernel, float, double) {}
