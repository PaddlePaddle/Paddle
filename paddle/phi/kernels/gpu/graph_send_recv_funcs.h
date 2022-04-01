// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/phi/kernels/graph_send_recv_kernel.h"

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <algorithm>
#include <vector>

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"

namespace phi {

template <typename T, typename IndexT>
struct GraphSendRecvSumCUDAFunctor {
  DEVICE inline void operator()(const T* params,
                                T* output,
                                const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicAdd(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT>
struct GraphSendRecvMaxCUDAFunctor {
  DEVICE inline void operator()(const T* params,
                                T* output,
                                const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicMax(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT>
struct GraphSendRecvMinCUDAFunctor {
  DEVICE inline void operator()(const T* params,
                                T* output,
                                const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicMin(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT, typename Functor>
__global__ void GraphSendRecvCUDAKernel(const T* params,
                                        const IndexT* src_indices,
                                        const IndexT* dst_indices,
                                        T* output,
                                        size_t index_size,
                                        size_t slice_size,
                                        Functor functor) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;
    IndexT src_i = src_indices[indices_i];
    IndexT dst_i = dst_indices[indices_i];
    int64_t in_i = src_i * slice_size + slice_i;
    int64_t out_i = dst_i * slice_size + slice_i;
    functor(params, output, in_i, out_i);
  }
}

// For max
template <typename T>
__global__ void InputResetMaxCUDAKernel(T* output,
                                        size_t input_size,
                                        size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, input_size * slice_size, int64_t) {
    if (*(output + i) == std::numeric_limits<T>::min()) {
      *(output + i) = 0;
    }
  }
}

// For min
template <typename T>
__global__ void InputResetMinCUDAKernel(T* output,
                                        size_t input_size,
                                        size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, input_size * slice_size, int64_t) {
    if (*(output + i) == std::numeric_limits<T>::max()) {
      *(output + i) = 0;
    }
  }
}

// Get dst_count
template <typename T, typename IndexT>
__global__ void ComputeCountCUDAKernel(int32_t* count,
                                       const IndexT* dst_indices,
                                       size_t index_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size, int64_t) {
    IndexT dst_i = dst_indices[i];
    paddle::platform::CudaAtomicAdd(count + dst_i, 1);
  }
}

// For forward mean
template <typename T>
__global__ void ManipulateMeanCUDAKernel(T* output,
                                         int32_t* count,
                                         size_t input_size,
                                         size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, input_size * slice_size, int64_t) {
    int64_t c_index = i / slice_size;
    if (*(count + c_index) > 1) {
      *(output + i) = *(output + i) / *(count + c_index);
    }
  }
}

// For backward mean
template <typename T, typename IndexT>
__global__ void ManipulateMeanGradCUDAKernel(const T* params,
                                             const IndexT* src_indices,
                                             const IndexT* dst_indices,
                                             T* output,
                                             size_t index_size,
                                             size_t slice_size,
                                             const int32_t* dst_count) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;
    IndexT src_i = src_indices[indices_i];
    IndexT dst_i = dst_indices[indices_i];
    int64_t in_i = src_i * slice_size + slice_i;
    int64_t out_i = dst_i * slice_size + slice_i;
    paddle::platform::CudaAtomicAdd(output + out_i,
                                    *(params + in_i) / dst_count[src_i]);
  }
}

// For backward min and max
template <typename T, typename IndexT>
__global__ void ManipulateMinMaxGradCUDAKernel(const T* params,
                                               const IndexT* src_indices,
                                               const IndexT* dst_indices,
                                               T* output,
                                               size_t index_size,
                                               size_t slice_size,
                                               const T* ptr_input,
                                               const T* ptr_output) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;
    IndexT src_i = src_indices[indices_i];
    IndexT dst_i = dst_indices[indices_i];
    int64_t in_i = src_i * slice_size + slice_i;
    int64_t out_i = dst_i * slice_size + slice_i;
    paddle::platform::CudaAtomicAdd(
        output + out_i,
        *(params + in_i) * (*(ptr_input + out_i) == *(ptr_output + in_i)));
  }
}

}  // namespace phi
