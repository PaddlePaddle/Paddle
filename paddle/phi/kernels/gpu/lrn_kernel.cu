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

#include "paddle/common/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/lrn_kernel_impl.h"
namespace phi {

template <typename T>
__global__ void KeCMRNormFillScale(int img_size,
                                   const T* in,
                                   T* mid,
                                   int C,
                                   int H,
                                   int W,
                                   int size,
                                   T k,
                                   T alpha,
                                   const DataLayout data_layout) {
  const int64_t idx =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x);
  if (idx < img_size) {
    const int64_t w = idx % W;
    const int64_t h = (idx / W) % H;
    const int64_t n = idx / W / H;
    const int64_t offset =
        (data_layout != DataLayout::NHWC ? (n * C * H + h) * W + w
                                         : ((n * H + h) * W + w) * C);

    in += offset;
    mid += offset;
    const int64_t step = static_cast<int64_t>(H) * W;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;

    T accum = 0;
    int64_t index = 0;
    while (index < C + post_pad) {
      if (index < C) {
        int64_t in_idx =
            (data_layout != DataLayout::NHWC ? index * step : index);
        T val = in[in_idx];
        accum += val * val;
      }
      if (index >= size) {
        int64_t in_idx =
            (data_layout != DataLayout::NHWC ? (index - size) * step
                                             : index - size);
        T val = in[in_idx];
        accum -= val * val;
      }
      if (index >= post_pad) {
        int64_t mid_idx =
            (data_layout != DataLayout::NHWC ? (index - post_pad) * step
                                             : index - post_pad);
        mid[mid_idx] = k + accum * alpha;
      }
      ++index;
    }
  }
}

template <typename T>
__global__ void KeCMRNormOutput(
    int input_size, const T* in, const T* mid, T negative_beta, T* out) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < input_size) {
    out[index] = in[index] * pow(mid[index], negative_beta);
  }
}

template <typename T>
void CrossMapNormal(const GPUContext& dev_ctx,
                    const T* inputs,
                    T* outputs,
                    T* mid,
                    int64_t N,
                    int64_t C,
                    int64_t H,
                    int64_t W,
                    int n,
                    T k,
                    T alpha,
                    T beta,
                    const DataLayout data_layout) {
  const int64_t img_size = N * H * W;
  const int64_t input_size = img_size * C;
  PADDLE_ENFORCE_LE_INT_MAX(img_size, "lrn img_size");
  PADDLE_ENFORCE_LE_INT_MAX(input_size, "lrn input_size");
  PADDLE_ENFORCE_LE_INT_MAX(C, "lrn C");
  PADDLE_ENFORCE_LE_INT_MAX(H, "lrn H");
  PADDLE_ENFORCE_LE_INT_MAX(W, "lrn W");

  const int block_size = 1024;
  const int64_t fill_grid_size = (img_size + block_size - 1) / block_size;
  PADDLE_ENFORCE_LE_UINT32_MAX(fill_grid_size, "lrn fill grid");
  const uint32_t fill_grid = static_cast<uint32_t>(fill_grid_size);

  KeCMRNormFillScale<T><<<fill_grid, block_size, 0, dev_ctx.stream()>>>(
      static_cast<int>(img_size),
      inputs,
      mid,
      static_cast<int>(C),
      static_cast<int>(H),
      static_cast<int>(W),
      n,
      k,
      alpha,
      data_layout);

  const int64_t output_grid_size = (input_size + block_size - 1) / block_size;
  PADDLE_ENFORCE_LE_UINT32_MAX(output_grid_size, "lrn output grid");
  const uint32_t output_grid = static_cast<uint32_t>(output_grid_size);
  KeCMRNormOutput<T><<<output_grid, block_size, 0, dev_ctx.stream()>>>(
      static_cast<int>(input_size), inputs, mid, -beta, outputs);
}

template <typename T>
struct LRNFunctor<GPUContext, T> {
  void operator()(const GPUContext& dev_ctx,
                  const DenseTensor& input,
                  DenseTensor* out,
                  DenseTensor* mid,
                  int64_t N,
                  int64_t C,
                  int64_t H,
                  int64_t W,
                  int n,
                  T k,
                  T alpha,
                  T beta,
                  const DataLayout data_layout) {
    CrossMapNormal<T>(dev_ctx,
                      input.data<T>(),
                      dev_ctx.Alloc<T>(out),
                      dev_ctx.Alloc<T>(mid),
                      N,
                      C,
                      H,
                      W,
                      n,
                      k,
                      alpha,
                      beta,
                      data_layout);
  }
};
template struct LRNFunctor<GPUContext, float>;
template struct LRNFunctor<GPUContext, double>;
}  // namespace phi

PD_REGISTER_KERNEL(lrn, GPU, ALL_LAYOUT, phi::LRNKernel, float) {}
