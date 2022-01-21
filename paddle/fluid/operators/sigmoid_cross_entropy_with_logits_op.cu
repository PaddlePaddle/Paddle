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
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/operators/sigmoid_cross_entropy_with_logits_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/pten/core/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#ifdef __HIPCC__
static constexpr int kNumCUDAThreads = 256;
#else
static constexpr int kNumCUDAThreads = 512;
#endif
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void GPUSigmoidForward(const T *x_data, const T *label_data,
                                  const int ignore_index, const int limit,
                                  T *out_data, T *counts) {
  CUDA_KERNEL_LOOP(i, limit) {
    T x = x_data[i];
    T label = label_data[i];
    T eps = static_cast<T>(1e-5);
    T diff = label - static_cast<T>(ignore_index);
    if ((diff > -eps) && (diff < eps)) {
      out_data[i] = static_cast<T>(0.);
      counts[i] = 0;
    } else {
      T term1 = (x > 0) ? x : 0;
      T term2 = x * label;
      T term3 = real_log(static_cast<T>(1) + real_exp(static_cast<T>(-abs(x))));
      out_data[i] = term1 - term2 + term3;
      counts[i] = 1;
    }
  }
}

template <typename T, int BlockDim>
__global__ void Sum(const T *counts, int num, const T eps, T *sum) {
  typedef cub::BlockReduce<double, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T in = 0;
  for (int i = threadIdx.x; i < num; i += BlockDim) {
    in += counts[i];
  }
  __syncthreads();
  auto out =
      BlockReduce(temp_storage).Reduce(static_cast<double>(in), cub::Sum());
  __syncthreads();
  if (threadIdx.x == 0) {
    T a = out > eps ? out : eps;
    sum[0] = a;
  }
}

template <typename T>
__global__ void Div(T *loss, const int num, const T *norm) {
  CUDA_KERNEL_LOOP(i, num) { loss[i] /= norm[0]; }
}

template <typename T>
__global__ void GPUSigmoidBackward(const T *x_data, const T *label_data,
                                   const int ignore_index, const T *dout_data,
                                   const int limit, T *dx_data, T *counts) {
  CUDA_KERNEL_LOOP(i, limit) {
    T x = x_data[i];
    T label = label_data[i];
    T dout = dout_data[i];
    T eps = static_cast<T>(1e-5);
    T diff = label - static_cast<T>(ignore_index);
    if ((diff > -eps) && (diff < eps)) {
      dx_data[i] = static_cast<T>(0.);
      counts[i] = 0;
    } else {
      T simoid_x = static_cast<T>(1) / (static_cast<T>(1) + real_exp(-x));
      T diff = simoid_x - label;
      dx_data[i] = dout * diff;
      counts[i] = 1;
    }
  }
}

// Out = max(X, 0) - X * Labels + log(1 + exp(-abs(X)))
template <typename DeviceContext, typename T>
class GPUSigmoidCrossEntropyWithLogitsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Label");
    Tensor *Out = context.Output<Tensor>("Out");
    int ignore_index = context.Attr<int>("ignore_index");
    auto out_data = Out->mutable_data<T>(context.GetPlace());

    auto &dev_ctx = context.cuda_device_context();
    bool normalize = context.Attr<bool>("normalize");

    // Temporary memory
    auto cnt_ptr = memory::Alloc(dev_ctx, Labels->numel() * sizeof(T));
    T *counts = reinterpret_cast<T *>(cnt_ptr->ptr());

    int limit = Out->numel();
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;
    GPUSigmoidForward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        X->data<T>(), Labels->data<T>(), ignore_index, limit, out_data, counts);
    if (normalize) {
      auto norm_ptr = memory::Alloc(dev_ctx, sizeof(T));
      T *norm = reinterpret_cast<T *>(norm_ptr->ptr());
      Sum<T, kNumCUDAThreads><<<1, kNumCUDAThreads, 0, dev_ctx.stream()>>>(
          counts, limit, static_cast<T>(1e-5), norm);
      Div<T><<<blocks, threads, 0, dev_ctx.stream()>>>(out_data, limit, norm);
    }
  }
};

// dX = sigmoid(X) - labels
template <typename DeviceContext, typename T>
class GPUSigmoidCrossEntropyWithLogitsGradKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Label");
    const Tensor *dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *dX = context.Output<Tensor>(framework::GradVarName("X"));
    auto dx_data = dX->mutable_data<T>(context.GetPlace());

    int ignore_index = context.Attr<int>("ignore_index");

    auto &dev_ctx = context.cuda_device_context();
    // Temporary memory
    auto cnt_ptr = memory::Alloc(dev_ctx, X->numel() * sizeof(T));
    T *counts = reinterpret_cast<T *>(cnt_ptr->ptr());

    int limit = dX->numel();
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;
    GPUSigmoidBackward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        X->data<T>(), Labels->data<T>(), ignore_index, dOut->data<T>(), limit,
        dx_data, counts);
    bool normalize = context.Attr<bool>("normalize");
    if (normalize) {
      auto norm_ptr = memory::Alloc(dev_ctx, sizeof(T));
      T *norm = reinterpret_cast<T *>(norm_ptr->ptr());
      Sum<T, kNumCUDAThreads><<<1, kNumCUDAThreads, 0, dev_ctx.stream()>>>(
          counts, limit, static_cast<T>(1e-5), norm);
      Div<T><<<blocks, threads, 0, dev_ctx.stream()>>>(dx_data, limit, norm);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(sigmoid_cross_entropy_with_logits,
                        ops::GPUSigmoidCrossEntropyWithLogitsKernel<
                            paddle::platform::CUDADeviceContext, float>,
                        ops::GPUSigmoidCrossEntropyWithLogitsKernel<
                            paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                        ops::GPUSigmoidCrossEntropyWithLogitsGradKernel<
                            paddle::platform::CUDADeviceContext, float>,
                        ops::GPUSigmoidCrossEntropyWithLogitsGradKernel<
                            paddle::platform::CUDADeviceContext, double>);
