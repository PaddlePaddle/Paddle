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

template <typename T, int BlockSize>
__global__ void SoftCrossEntropyKernel(T* Y, const T* X, const T* label,
                                       const int N, const int D) {
  int tid = threadIdx.x;
  __shared__ T d_sum[BlockSize];
  int next_idx = blockIdx.x * D + tid;

  d_sum[tid] = 0;
  int cur_idx = tid;
  while (cur_idx < D) {
    d_sum[tid] += TolerableValue<T>()(std::log(X[next_idx])) * label[next_idx];
    next_idx += BlockSize;
    cur_idx += BlockSize;
  }
  __syncthreads();

  for (int stride = BlockSize >> 1; stride > 0; stride >>= 1) {
    __syncthreads();
    if (tid < stride) {
      next_idx = tid + stride;
      d_sum[tid] += d_sum[next_idx];
    }
  }
  __syncthreads();

  if (tid == 0) {
    Y[blockIdx.x] = -d_sum[0];
  }
}

// TODO(qingqing): make zero setting an common function.
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
  int row_ids = blockIdx.x * blockDim.x + threadIdx.x;
  int col_ids = blockIdx.y * blockDim.y + threadIdx.y;
  int ids = row_ids * D + col_ids;

  if (ids < N * D) {
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

    int n = x->dims()[0];
    int d = x->dims()[1];
    int block = 512;
    int grid = (n + block - 1) / block;
    // TODO(qingqing) launch kernel on specified stream
    // base on ExecutionContext.
    if (ctx.Attr<int>("soft_label") == 1) {
      auto* label_data = ctx.Input<Tensor>("Label")->data<T>();
      grid = d;
      SoftCrossEntropyKernel<T, 512><<<grid, block>>>(y_data, x_data,
                                                      label_data, n, d);
    } else {
      auto* label_data = ctx.Input<Tensor>("Label")->data<int>();
      CrossEntropyKernel<T><<<grid, block>>>(y_data, x_data, label_data, n, d);
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
    zero<T><<<grid, block>>>(dx_data, n * d);
    grid = (n + block - 1) / block;
    // TODO(qingqing): launch kernel on specified stream
    // base on ExecutionContext.
    if (ctx.Attr<int>("soft_label") == 1) {
      int block_x = 32;
      int block_y = 32;
      dim3 block(block_x, block_y);
      dim3 grid((n + block_x - 1) / block_x, (d + block_y - 1) / block_y);

      auto* label_data = label->data<T>();
      SoftCrossEntropyGradientKernel<T><<<grid, block>>>(
          dx_data, dy_data, x_data, label_data, n, d);
    } else {
      auto* label_data = label->data<int>();
      CrossEntropyGradientKernel<T><<<grid, block>>>(dx_data, dy_data, x_data,
                                                     label_data, n, d);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(cross_entropy, ops::CrossEntropyOpCUDAKernel<float>);
REGISTER_OP_GPU_KERNEL(cross_entropy_grad,
                       ops::CrossEntropyGradientOpCUDAKernel<float>);
