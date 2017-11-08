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

#include "paddle/operators/math/cross_entropy.h"

namespace paddle {
namespace operators {
namespace math {

namespace {
template <typename T>
__global__ void CrossEntropyKernel(T* Y, const T* X, const int64_t* label,
                                   const int N, const int D) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    PADDLE_ASSERT(label[i] >= 0 && label[i] < D);
    Y[i] = -math::TolerableValue<T>()(log(X[i * D + label[i]]));
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

// CUDA do not support dynamic arrary in template
// https://stackoverflow.com/questions/20497209
template <typename T>
struct SharedMemory {
  // Ensure that we won't compile any un-specialized types
  __device__ T* GetPointer() { return NULL; }
};

template <>
struct SharedMemory<float> {
  __device__ float* GetPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <>
struct SharedMemory<double> {
  __device__ double* GetPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

template <typename T>
__global__ void SoftCrossEntropyKernel(T* Y, const T* X, const T* label,
                                       const int class_num) {
  int tid = threadIdx.x;
  SharedMemory<T> d_sum_shared;
  T* d_sum = d_sum_shared.GetPointer();
  d_sum[tid] = 0;

  int cur_idx = tid;
  int next_idx = blockIdx.x * class_num + tid;
  while (cur_idx < class_num) {
    d_sum[tid] +=
        math::TolerableValue<T>()(std::log(X[next_idx])) * label[next_idx];
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
}  // namespace

using Tensor = framework::Tensor;

template <typename T>
class CrossEntropyFunctor<platform::GPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& ctx, framework::Tensor* out,
                  const framework::Tensor* prob,
                  const framework::Tensor* labels, bool softLabel) {
    const T* prob_data = prob->data<T>();
    T* loss_data = out->mutable_data<T>(ctx.GetPlace());

    int batch_size = prob->dims()[0];
    int class_num = prob->dims()[1];

    if (softLabel) {
      const T* label_data = labels->data<T>();
      int block = class_num > 512 ? 512 : pow(2, int(std::log2(class_num)));

      SoftCrossEntropyKernel<T><<<
          batch_size, block, block * sizeof(T),
          reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
          loss_data, prob_data, label_data, class_num);
    } else {
      const int64_t* label_data = labels->data<int64_t>();
      int block = 512;
      int grid = (batch_size + block - 1) / block;
      CrossEntropyKernel<T><<<
          grid, block, 0,
          reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
          loss_data, prob_data, label_data, batch_size, class_num);
    }
  }
};

template class CrossEntropyFunctor<platform::GPUPlace, float>;
template class CrossEntropyFunctor<platform::GPUPlace, double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
