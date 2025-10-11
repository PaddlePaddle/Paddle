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

#include "paddle/phi/kernels/gpu/correlation_grad_kernel.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/correlation_funcs.cu.h"

namespace phi {

template <typename T>
__global__ void correlation_backward_input1(int64_t n,
                                            T *grad_input1,
                                            const int64_t input_channel,
                                            const int64_t input_height,
                                            const int64_t input_width,
                                            const T *grad_output,
                                            const int64_t output_channel,
                                            const int64_t output_height,
                                            const int64_t output_width,
                                            const T *rinput2,
                                            const int pad_size,
                                            const int kernel_size,
                                            const int max_displacement,
                                            const int stride1,
                                            const int stride2) {
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total_hw_c = input_channel * input_height * input_width;
  if (thread_index >= total_hw_c) return;

  int64_t c = thread_index / (input_height * input_width);
  int64_t hw_index = thread_index % (input_height * input_width);
  int64_t h = hw_index / input_width + pad_size;
  int64_t w = hw_index % input_width + pad_size;

  int kernel_rad = (kernel_size - 1) / 2;
  int displacement_rad = max_displacement / stride2;
  int displacement_size = 2 * displacement_rad + 1;

  int64_t xmin = (w - kernel_rad - max_displacement) / stride1;
  int64_t ymin = (h - kernel_rad - max_displacement) / stride1;
  int64_t xmax = (w + kernel_rad - max_displacement) / stride1;
  int64_t ymax = (h + kernel_rad - max_displacement) / stride1;

  if (xmax < 0 || ymax < 0 || xmin >= output_width || ymin >= output_height)
    return;
  if (xmin > xmax || ymin > ymax) return;

  xmin = max(static_cast<int64_t>(0), xmin);
  xmax = min(output_width - 1, xmax);
  ymin = max(static_cast<int64_t>(0), ymin);
  ymax = min(output_height - 1, ymax);

  int64_t p_input_width = input_width + 2 * pad_size;
  int64_t p_input_height = input_height + 2 * pad_size;
  int64_t p_dimchw = input_channel * p_input_height * p_input_width;
  int64_t p_dimcw = input_channel * p_input_width;
  int64_t p_dimc = input_channel;

  int64_t t_dimchw = output_channel * output_height * output_width;
  int64_t t_dimhw = output_height * output_width;
  int64_t t_dimw = output_width;

  int64_t o_dimchw = input_channel * input_height * input_width;
  int64_t o_dimhw = input_height * input_width;
  int64_t o_dimw = input_width;

  int64_t nelems = kernel_size * kernel_size * input_channel;

  T sum = 0;

  for (int64_t tc = 0; tc < output_channel; ++tc) {
    int64_t i2 = (tc % displacement_size - displacement_rad) * stride2;
    int64_t j2 = (tc / displacement_size - displacement_rad) * stride2;

    int64_t index2 = n * p_dimchw + (h + j2) * p_dimcw + (w + i2) * p_dimc + c;
    T val2 = rinput2[index2];

    for (int j = ymin; j <= ymax; ++j) {
      for (int i = xmin; i <= xmax; ++i) {
        int64_t t_index = n * t_dimchw + tc * t_dimhw + j * t_dimw + i;
        sum += grad_output[t_index] * val2;
      }
    }
  }

  const int64_t index1 =
      n * o_dimchw + c * o_dimhw + (h - pad_size) * o_dimw + (w - pad_size);
  grad_input1[index1] = sum / nelems;
}

template <typename T>
__global__ void correlation_backward_input2(int64_t n,
                                            T *grad_input2,
                                            const int64_t input_channel,
                                            const int64_t input_height,
                                            const int64_t input_width,
                                            const T *grad_output,
                                            const int64_t output_channel,
                                            const int64_t output_height,
                                            const int64_t output_width,
                                            const T *rinput1,
                                            const int pad_size,
                                            const int kernel_size,
                                            const int max_displacement,
                                            const int stride1,
                                            const int stride2) {
  int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total_hw_c = input_channel * input_height * input_width;
  if (thread_index >= total_hw_c) return;

  int64_t c = thread_index / (input_height * input_width);
  int64_t hw_index = thread_index % (input_height * input_width);
  int64_t h = hw_index / input_width + pad_size;
  int64_t w = hw_index % input_width + pad_size;

  int kernel_rad = (kernel_size - 1) / 2;
  int displacement_rad = max_displacement / stride2;
  int displacement_size = 2 * displacement_rad + 1;

  int64_t p_input_width = input_width + 2 * pad_size;
  int64_t p_input_height = input_height + 2 * pad_size;
  int64_t p_dimchw = input_channel * p_input_height * p_input_width;
  int64_t p_dimcw = input_channel * p_input_width;
  int64_t p_dimc = input_channel;

  int64_t t_dimchw = output_channel * output_height * output_width;
  int64_t t_dimhw = output_height * output_width;
  int64_t t_dimw = output_width;

  int64_t o_dimchw = input_channel * input_height * input_width;
  int64_t o_dimhw = input_height * input_width;
  int64_t o_dimw = input_width;

  int64_t nelems = kernel_size * kernel_size * input_channel;

  T sum = 0;

  for (int64_t tc = 0; tc < output_channel; ++tc) {
    int64_t i2 = (tc % displacement_size - displacement_rad) * stride2;
    int64_t j2 = (tc / displacement_size - displacement_rad) * stride2;

    int64_t xmin = (w - kernel_rad - max_displacement - i2) / stride1;
    int64_t ymin = (h - kernel_rad - max_displacement - j2) / stride1;
    int64_t xmax = (w + kernel_rad - max_displacement - i2) / stride1;
    int64_t ymax = (h + kernel_rad - max_displacement - j2) / stride1;

    if (xmax < 0 || ymax < 0 || xmin >= output_width || ymin >= output_height)
      continue;
    if (xmin > xmax || ymin > ymax) continue;

    xmin = max(static_cast<int64_t>(0), xmin);
    xmax = min(output_width - 1, xmax);
    ymin = max(static_cast<int64_t>(0), ymin);
    ymax = min(output_height - 1, ymax);

    int64_t index1 = n * p_dimchw + (h - j2) * p_dimcw + (w - i2) * p_dimc + c;
    T val1 = rinput1[index1];

    for (int j = ymin; j <= ymax; ++j) {
      for (int i = xmin; i <= xmax; ++i) {
        int64_t t_index = n * t_dimchw + tc * t_dimhw + j * t_dimw + i;
        sum += grad_output[t_index] * val1;
      }
    }
  }

  const int64_t index2 =
      n * o_dimchw + c * o_dimhw + (h - pad_size) * o_dimw + (w - pad_size);
  grad_input2[index2] = sum / nelems;
}

template <typename T, typename Context>
void CorrelationCUDAGradKernel(const Context &dev_ctx,
                               const DenseTensor &input1,
                               const DenseTensor &input2,
                               const DenseTensor &out_grad,
                               int pad_size,
                               int kernel_size,
                               int max_displacement,
                               int stride1,
                               int stride2,
                               int corr_type_multiply,
                               DenseTensor *input1_grad,
                               DenseTensor *input2_grad) {
  const auto *grad_output = &out_grad;

  auto *grad_input1 = input1_grad;
  dev_ctx.template Alloc<T>(grad_input1);
  auto *grad_input2 = input2_grad;
  dev_ctx.template Alloc<T>(grad_input2);

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

  set_zero<<<static_cast<int64_t>(grid_size), 512, 0, dev_ctx.stream()>>>(
      rinput1.data<T>(), rinput1.numel());
  grid_size = std::min(static_cast<int64_t>((rinput2.numel() + 512 - 1) / 512),
                       max_grid_dim);
  set_zero<<<grid_size, 512, 0, dev_ctx.stream()>>>(rinput2.data<T>(),
                                                    rinput2.numel());
  grid_size =
      std::min(static_cast<int64_t>((grad_input1->numel() + 512 - 1) / 512),
               max_grid_dim);
  set_zero<<<grid_size, 512, 0, dev_ctx.stream()>>>(grad_input1->data<T>(),
                                                    grad_input1->numel());
  grid_size =
      std::min(static_cast<int64_t>((grad_input2->numel() + 512 - 1) / 512),
               max_grid_dim);
  set_zero<<<grid_size, 512, 0, dev_ctx.stream()>>>(grad_input2->data<T>(),
                                                    grad_input2->numel());

  auto grad_out_dims = grad_output->dims();
  int GOC = grad_out_dims[1];
  int GOH = grad_out_dims[2];
  int GOW = grad_out_dims[3];

  int blocks_grid = std::min(static_cast<int64_t>(N) * H * W, max_grid_dim);
  dim3 threads_block(THREADS_PER_BLOCK);

  channel_first<T><<<blocks_grid, threads_block, 0, dev_ctx.stream()>>>(
      input1.data<T>(), rinput1.data<T>(), N, C, H, W, pad_size);
  channel_first<T><<<blocks_grid, threads_block, 0, dev_ctx.stream()>>>(
      input2.data<T>(), rinput2.data<T>(), N, C, H, W, pad_size);

  dim3 threadsPerBlock(THREADS_PER_BLOCK);
  dim3 totalBlocksCorr(H, W, C);
  grid_size =
      std::min((static_cast<int64_t>(C) * H * W + THREADS_PER_BLOCK - 1) /
                   THREADS_PER_BLOCK,
               max_grid_dim);

  for (int n = 0; n < N; n++) {
    correlation_backward_input1<T>
        <<<grid_size, threadsPerBlock, 0, dev_ctx.stream()>>>(
            n,
            grad_input1->data<T>(),
            C,
            H,
            W,
            grad_output->data<T>(),
            GOC,
            GOH,
            GOW,
            rinput2.data<T>(),
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2);
  }

  for (int n = 0; n < N; n++) {
    correlation_backward_input2<T>
        <<<grid_size, threadsPerBlock, 0, dev_ctx.stream()>>>(
            n,
            grad_input2->data<T>(),
            C,
            H,
            W,
            grad_output->data<T>(),
            GOC,
            GOH,
            GOW,
            rinput1.data<T>(),
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(correlation_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CorrelationCUDAGradKernel,
                   float,
                   double) {}
