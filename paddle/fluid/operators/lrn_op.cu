/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/lrn_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void KeCMRNormFillScale(int img_size, const T* in, T* mid, int C,
                                   int H, int W, int size, T k, T alpha) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < img_size) {
    const int w = idx % W;
    const int h = (idx / W) % H;
    const int n = idx / W / H;
    const int offset = (n * C * H + h) * W + w;

    in += offset;
    mid += offset;
    const int step = H * W;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;

    T accum = 0;
    int index = 0;
    while (index < C + post_pad) {
      if (index < C) {
        T val = in[index * step];
        accum += val * val;
      }
      if (index >= size) {
        T val = in[(index - size) * step];
        accum -= val * val;
      }
      if (index >= post_pad) {
        mid[(index - post_pad) * step] = k + accum * alpha;
      }
      ++index;
    }
  }
}

template <typename T>
__global__ void KeCMRNormOutput(int input_size, const T* in, const T* mid,
                                T negative_beta, T* out) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < input_size) {
    out[index] = in[index] * pow(mid[index], negative_beta);
  }
}

template <typename T>
void CrossMapNormal(const framework::ExecutionContext& ctx, const T* inputs,
                    T* outputs, T* mid, int N, int C, int H, int W, int n, T k,
                    T alpha, T beta) {
  int img_size = N * H * W;
  const int block_size = 1024;
  int grid_size = (img_size + block_size - 1) / block_size;

  auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  KeCMRNormFillScale<T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      img_size, inputs, mid, C, H, W, n, k, alpha);

  int input_size = N * H * W * C;
  grid_size = (input_size + block_size - 1) / block_size;
  KeCMRNormOutput<T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      input_size, inputs, mid, -beta, outputs);
}

template <typename T>
struct LRNFunctor<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor& input, framework::Tensor* out,
                  framework::Tensor* mid, int N, int C, int H, int W, int n,
                  T k, T alpha, T beta) {
    CrossMapNormal<T>(
        ctx, input.data<T>(), out->mutable_data<T>(ctx.GetPlace()),
        mid->mutable_data<T>(ctx.GetPlace()), N, C, H, W, n, k, alpha, beta);
  }
};

template struct LRNFunctor<platform::CUDADeviceContext, float>;
template struct LRNFunctor<platform::CUDADeviceContext, double>;

template <typename T>
__global__ void KeCMRNormDiff(int img_size, const T* x, const T* out,
                              const T* mid, T* x_g, const T* out_g, int C,
                              int H, int W, int size, T negative_beta,
                              T ratio) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < img_size) {
    const int w = idx % W;
    const int h = (idx / W) % H;
    const int n = idx / W / H;
    const int offset = (n * C * H + h) * W + w;
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
        x_g[index * step] = 0.0;
        accum += out_g[index * step] * out[index * step] / mid[index * step];
      }
      if (index >= size) {
        accum -= out_g[(index - size) * step] * out[(index - size) * step] /
                 mid[(index - size) * step];
      }
      if (index >= post_pad) {
        x_g[(index - post_pad) * step] +=
            out_g[(index - post_pad) * step] *
                pow(mid[(index - post_pad) * step], negative_beta) -
            ratio * x[(index - post_pad) * step] * accum;
      }
      ++index;
    }
  }
}

template <typename T>
void CrossMapNormalGrad(const framework::ExecutionContext& ctx, const T* x,
                        const T* out, const T* mid, T* x_g, const T* out_g,
                        int N, int C, int H, int W, int n, T alpha, T beta) {
  int img_size = N * H * W;

  const int block_size = 1024;
  int grid_size = (img_size + block_size - 1) / block_size;

  auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  KeCMRNormDiff<T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      img_size, x, out, mid, x_g, out_g, C, H, W, n, -beta,
      2.0f * alpha * beta);
}

template <typename T>
struct LRNGradFunctor<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor& x, const framework::Tensor& out,
                  const framework::Tensor& mid, framework::Tensor* x_g,
                  const framework::Tensor& out_g, int N, int C, int H, int W,
                  int n, T alpha, T beta) {
    CrossMapNormalGrad<T>(ctx, x.data<T>(), out.data<T>(), mid.data<T>(),
                          x_g->mutable_data<T>(ctx.GetPlace()), out_g.data<T>(),
                          N, C, H, W, n, alpha, beta);
  }
};

template struct LRNGradFunctor<platform::CUDADeviceContext, float>;
template struct LRNGradFunctor<platform::CUDADeviceContext, double>;
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    lrn, ops::LRNKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    lrn_grad, ops::LRNGradKernel<paddle::platform::CUDADeviceContext, float>);
