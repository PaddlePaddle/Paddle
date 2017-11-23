/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define EIGEN_USE_GPU
#include "paddle/operators/lrn_op.h"

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
        accum += in[index * step] * in[index * step];
      }
      if (index >= size) {
        accum -= in[(index - size) * step] * in[(index - size) * step];
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
  int block_size = 1024;
  int grid_size = (img_size + 1024 - 1) / 1024;

  const auto& stream =
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx.device_context())
          .stream();
  KeCMRNormFillScale<T><<<grid_size, block_size, 0, stream>>>(
      img_size, inputs, mid, C, H, W, n, k, alpha);

  int input_size = N * H * W * C;
  block_size = 1024;
  grid_size = (input_size + 1024 - 1) / 1024;
  KeCMRNormOutput<T><<<grid_size, block_size, 0, stream>>>(input_size, inputs,
                                                           mid, -beta, outputs);
}

template <typename T>
struct LRNFunctor<platform::GPUPlace, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* input, framework::Tensor* mid,
                  framework::Tensor* out, int N, int C, int H, int W, int n,
                  T k, T alpha, T beta) {
    CrossMapNormal<T>(
        ctx, input->data<T>(), out->mutable_data<T>(ctx.GetPlace()),
        mid->mutable_data<T>(ctx.GetPlace()), N, C, H, W, n, k, alpha, beta);
  }
};

template struct LRNFunctor<platform::GPUPlace, float>;
template struct LRNFunctor<platform::GPUPlace, double>;
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(lrn, ops::LRNKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(lrn_grad,
                       ops::LRNGradKernel<paddle::platform::GPUPlace, float>);
