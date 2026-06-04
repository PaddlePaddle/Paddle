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
__global__ void KeCMRNormDiff(int img_size,
                              const T* x,
                              const T* out,
                              const T* mid,
                              T* x_g,
                              const T* out_g,
                              int C,
                              int H,
                              int W,
                              int size,
                              T negative_beta,
                              T ratio,
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
    x += offset;
    out += offset;
    mid += offset;
    out_g += offset;
    x_g += offset;

    const int64_t step = static_cast<int64_t>(H) * W;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;

    int64_t index = 0;
    T accum = 0;
    // TODO(gongwb): optimize this with thread shared array.
    while (index < C + post_pad) {
      if (index < C) {
        int64_t idx_val =
            (data_layout != DataLayout::NHWC ? index * step : index);
        x_g[idx_val] = 0.0;
        accum += out_g[idx_val] * out[idx_val] / mid[idx_val];
      }
      if (index >= size) {
        int64_t idx_val =
            (data_layout != DataLayout::NHWC ? (index - size) * step
                                             : index - size);
        accum -= out_g[idx_val] * out[idx_val] / mid[idx_val];
      }
      if (index >= post_pad) {
        int64_t idx_val =
            (data_layout != DataLayout::NHWC ? (index - post_pad) * step
                                             : index - post_pad);
        x_g[idx_val] += out_g[idx_val] * pow(mid[idx_val], negative_beta) -
                        ratio * x[idx_val] * accum;
      }
      ++index;
    }
  }
}

template <typename T>
void CrossMapNormalGrad(const GPUContext& dev_ctx,
                        const T* x,
                        const T* out,
                        const T* mid,
                        T* x_g,
                        const T* out_g,
                        int64_t N,
                        int64_t C,
                        int64_t H,
                        int64_t W,
                        int n,
                        T alpha,
                        T beta,
                        const DataLayout data_layout) {
  int64_t img_size = N * H * W;

  const int block_size = 1024;
  int64_t grid_size = (img_size + block_size - 1) / block_size;
  PADDLE_ENFORCE_LE_INT_MAX(grid_size, "grid_size");
  PADDLE_ENFORCE_LE_INT_MAX(C, "C");

  KeCMRNormDiff<T>
      <<<static_cast<int>(grid_size), block_size, 0, dev_ctx.stream()>>>(
          static_cast<int>(img_size),
          x,
          out,
          mid,
          x_g,
          out_g,
          static_cast<int>(C),
          static_cast<int>(H),
          static_cast<int>(W),
          n,
          -beta,
          2.0f * alpha * beta,
          data_layout);
}

template <typename T>
struct LRNGradFunctor<GPUContext, T> {
  void operator()(const GPUContext& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& out,
                  const DenseTensor& mid,
                  DenseTensor* x_g,
                  const DenseTensor& out_g,
                  int64_t N,
                  int64_t C,
                  int64_t H,
                  int64_t W,
                  int n,
                  T alpha,
                  T beta,
                  const DataLayout data_layout) {
    CrossMapNormalGrad<T>(dev_ctx,
                          x.data<T>(),
                          out.data<T>(),
                          mid.data<T>(),
                          dev_ctx.Alloc<T>(x_g),
                          out_g.data<T>(),
                          N,
                          C,
                          H,
                          W,
                          n,
                          alpha,
                          beta,
                          data_layout);
  }
};

template struct LRNGradFunctor<GPUContext, float>;
template struct LRNGradFunctor<GPUContext, double>;
}  // namespace phi

PD_REGISTER_KERNEL(lrn_grad, GPU, ALL_LAYOUT, phi::LRNGradKernel, float) {}
