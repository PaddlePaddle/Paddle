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

#include "paddle/framework/op_registry.h"
#include "paddle/operators/cross_entropy_op.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/hostdevice.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void CrossEntropyKernel(T* Y, const T* X, const int* label,
                                   const int N, const int D) {
  // TOOD(qingqing) define CUDA_1D_KERNEL_LOOP macro in a common file.
  // CUDA_1D_KERNEL_LOOP(i, N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    PADDLE_ASSERT(label[i] >= 0 && label[i] < D);
    Y[i] = -TolerableValue<T>()(log(X[i * D + label[i]]));
  }
}

template <typename T>
__device__ __forceinline__ T sum_single_warp(T val) {
  val += __shfl_down(val, 16);
  val += __shfl_down(val, 8);
  val += __shfl_down(val, 4);
  val += __shfl_down(val, 2);
  val += __shfl_down(val, 1);
  return val;
}

template <typename T>
__global__ void SoftCrossEntropyKernel(T* Y, const T* X, const T* label,
                                       const int class_num) {
  int tid = threadIdx.x;
  extern __shared__ T d_sum[];
  d_sum[tid] = 0;

  int cur_idx = tid;
  int next_idx = blockIdx.x * class_num + tid;
  while (cur_idx < class_num) {
    d_sum[tid] += TolerableValue<T>()(std::log(X[next_idx])) * label[next_idx];
    next_idx += blockDim.x;
    cur_idx += blockDim.x;
  }
  __syncthreads();

  for (unsigned int stride = blockDim.x >> 1; stride >= 32; stride >>= 1) {
    if (tid < stride) d_sum[tid] += d_sum[tid + stride];
    __syncthreads();
  }

  T val = d_sum[tid];
  val = sum_single_warp<T>(val);
  if (tid == 0) Y[blockIdx.x] = -val;
}

// TODO(qingqing): make zero setting a common function.
template <typename T>
__global__ void zero(T* X, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    X[i] = 0.0;
  }
}

template <typename T>
__global__ void CrossEntropyGradientKernel(T* dX, const T* dY, const T* X,
                                           const int* label, const int N,
                                           const int D) {
  // TOOD(qingqing) define CUDA_1D_KERNEL_LOOP macro in a common file.
  // CUDA_1D_KERNEL_LOOP(i, N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    int idx = i * D + label[i];
    dX[idx] = -dY[i] / X[idx];
  }
}

template <typename T>
__global__ void SoftCrossEntropyGradientKernel(T* dX, const T* dY, const T* X,
                                               const T* label, const int N,
                                               const int D) {
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < N * D) {
    int row_ids = ids / D;
    dX[ids] = -label[ids] * dY[row_ids] / X[ids];
  }
}

template <typename T>
class CrossEntropyOpCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");

    auto x = ctx.Input<Tensor>("X");
    auto y = ctx.Output<Tensor>("Y");
    auto label = ctx.Input<Tensor>("Label");

    auto* x_data = x->data<T>();
    y->mutable_data<T>(ctx.GetPlace());
    auto* y_data = y->data<T>();

    int batch_size = x->dims()[0];
    int class_num = x->dims()[1];

    if (ctx.Attr<bool>("soft_label")) {
      auto* label_data = ctx.Input<Tensor>("Label")->data<T>();
      int block = class_num > 512 ? 512 : pow(2, int(std::log2(class_num)));

      SoftCrossEntropyKernel<
          T><<<batch_size, block, block * sizeof(T),
               reinterpret_cast<const platform::CUDADeviceContext&>(
                   ctx.device_context())
                   .stream()>>>(y_data, x_data, label_data, class_num);
    } else {
      auto* label_data = ctx.Input<Tensor>("Label")->data<int>();
      int block = 512;
      int grid = (batch_size + block - 1) / block;
      CrossEntropyKernel<T><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(y_data, x_data, label_data,
                                           batch_size, class_num);
    }
  }
};

template <typename T>
class CrossEntropyGradientOpCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");

    auto x = ctx.Input<Tensor>("X");
    auto dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto label = ctx.Input<Tensor>("Label");

    auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    auto* dy_data = dy->data<T>();
    auto* x_data = x->data<T>();

    int n = x->dims()[0];
    int d = x->dims()[1];

    int block = 512;
    int grid = (n * d + block - 1) / block;
    zero<T><<<grid, block, 0,
              reinterpret_cast<const platform::CUDADeviceContext&>(
                  ctx.device_context())
                  .stream()>>>(dx_data, n * d);
    if (ctx.Attr<bool>("soft_label")) {
      auto* label_data = label->data<T>();
      SoftCrossEntropyGradientKernel<T><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(dx_data, dy_data, x_data, label_data,
                                           n, d);
    } else {
      auto* label_data = label->data<int>();
      CrossEntropyGradientKernel<T><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(dx_data, dy_data, x_data, label_data,
                                           n, d);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(cross_entropy, ops::CrossEntropyOpCUDAKernel<float>);
REGISTER_OP_GPU_KERNEL(cross_entropy_grad,
                       ops::CrossEntropyGradientOpCUDAKernel<float>);
