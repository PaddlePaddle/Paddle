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
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < img_size) {
    const int w = idx % W;
    const int h = (idx / W) % H;
    const int n = idx / W / H;
    const int offset =
        (data_layout != DataLayout::kNHWC ? (n * C * H + h) * W + w
                                          : ((n * H + h) * W + w) * C);
    x += offset;
    out += offset;
    mid += offset;
    out_g += offset;
    x_g += offset;

    const int step = H * W;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;

    int index = 0;
    T accum = 0;
    // TODO(gongwb): optimize this with thread shared array.
    while (index < C + post_pad) {
      if (index < C) {
        int idx = (data_layout != DataLayout::kNHWC ? index * step : index);
        x_g[idx] = 0.0;
        accum += out_g[idx] * out[idx] / mid[idx];
      }
      if (index >= size) {
        int idx = (data_layout != DataLayout::kNHWC ? (index - size) * step
                                                    : index - size);
        accum -= out_g[idx] * out[idx] / mid[idx];
      }
      if (index >= post_pad) {
        int idx = (data_layout != DataLayout::kNHWC ? (index - post_pad) * step
                                                    : index - post_pad);
        x_g[idx] +=
            out_g[idx] * pow(mid[idx], negative_beta) - ratio * x[idx] * accum;
      }
      ++index;
    }
  }
}

template <typename T>
void CrossMapNormalGrad(const phi::GPUContext& dev_ctx,
                        const T* x,
                        const T* out,
                        const T* mid,
                        T* x_g,
                        const T* out_g,
                        int N,
                        int C,
                        int H,
                        int W,
                        int n,
                        T alpha,
                        T beta,
                        const DataLayout data_layout) {
  int img_size = N * H * W;

  const int block_size = 1024;
  int grid_size = (img_size + block_size - 1) / block_size;

  KeCMRNormDiff<T>
      <<<grid_size, block_size, 0, dev_ctx.stream()>>>(img_size,
                                                       x,
                                                       out,
                                                       mid,
                                                       x_g,
                                                       out_g,
                                                       C,
                                                       H,
                                                       W,
                                                       n,
                                                       -beta,
                                                       2.0f * alpha * beta,
                                                       data_layout);
}

template <typename T>
struct LRNGradFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& out,
                  const phi::DenseTensor& mid,
                  phi::DenseTensor* x_g,
                  const phi::DenseTensor& out_g,
                  int N,
                  int C,
                  int H,
                  int W,
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

template struct LRNGradFunctor<phi::GPUContext, float>;
template struct LRNGradFunctor<phi::GPUContext, double>;
}  // namespace phi

PD_REGISTER_KERNEL(lrn_grad, GPU, ALL_LAYOUT, phi::LRNGradKernel, float) {}
