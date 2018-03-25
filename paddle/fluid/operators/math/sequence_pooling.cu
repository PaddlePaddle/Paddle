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

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sequence_pooling.h"

#include "paddle/fluid/platform/cuda_helper.h"

namespace paddle {
namespace operators {
namespace math {

#define FLT_MAX __FLT_MAX__

template <typename T>
struct MaxPoolFunctor {
  HOSTDEVICE void operator()(const T* input, const size_t start,
                             const size_t end, const size_t item_dim, T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      T max_val = static_cast<T>(FLT_MAX);
      int max_index = -1;
      for (int i = start; i < end; ++i) {
        if (max_val < input[item_dim * i + tid]) {
          max_val = input[item_dim * i + tid];
          max_index = i;
        }
      }
      output[tid] = max_val;
      index[tid] = max_index;
    }
  }
};

template <typename T>
struct AvgPoolFunctor {
  HOSTDEVICE void operator()(const T* input, const size_t start,
                             const size_t end, const size_t item_dim, T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      T val = static_cast<T>(0);
      for (int i = start; i < end; ++i) {
        val += input[item_dim * i + tid];
      }
      // end, start is lod, so end - start != 0
      output[tid] = val / (end - start);
    }
  }
};

template <typename T>
struct SumPoolFunctor {
  HOSTDEVICE void operator()(const T* input, const size_t start,
                             const size_t end, const size_t item_dim, T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      T val = static_cast<T>(0);
      for (int i = start; i < end; ++i) {
        val += input[item_dim * i + tid];
      }
      output[tid] = val;
    }
  }
};

template <typename T>
struct SqrtPoolFunctor {
  HOSTDEVICE void operator()(const T* input, const size_t start,
                             const size_t end, const size_t item_dim, T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      T val = static_cast<T>(0);
      for (int i = start; i < end; ++i) {
        val += input[item_dim * i + tid];
      }
      // end, start is lod, so end - start != 0
      output[tid] = val / sqrt(end - start);
    }
  }
};

template <typename T>
struct LastPoolFunctor {
  HOSTDEVICE void operator()(const T* input, const size_t start,
                             const size_t end, const size_t item_dim, T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        if (i == end - 1) {
          output[tid] = input[item_dim * i + tid];
        } else {
          output[tid] = static_cast<T>(0);
        }
      }
    }
  }
};

template <typename T>
struct FirstPoolFunctor {
  HOSTDEVICE void operator()(const T* input, const size_t start,
                             const size_t end, const size_t item_dim, T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        if (i == start) {
          output[tid] = input[item_dim * i + tid];
        } else {
          output[tid] = static_cast<T>(0);
        }
      }
    }
  }
};

template <typename T, typename Range_OP>
__global__ void sequence_pool_kernel(Range_OP op, const T* input,
                                     const size_t* lod, const size_t lod_size,
                                     const size_t item_dim, T* output,
                                     int* index) {
  int bid = blockIdx.x;
  if (bid >= lod_size) return;
  size_t start = lod[bid];
  size_t end = lod[bid + 1];
  op(input, start, end, item_dim, &output[bid * item_dim], index);
}

template <typename T>
class SequencePoolFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const std::string pooltype, const framework::LoDTensor& input,
                  framework::Tensor* output,
                  framework::Tensor* index = nullptr) {
    auto lod = input.lod()[0];
    int item_dim = input.numel() / output->dims()[0];
    dim3 threads(1024, 1);
    dim3 grid(lod.size(), 1);
    if (pooltype == "MAX") {
      sequence_pool_kernel<
          T, MaxPoolFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          MaxPoolFunctor<T>(), input.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          output->mutable_data<T>(context.GetPlace()), index->data<T>());
    } else if (pooltype == "AVG") {
      sequence_pool_kernel<
          T, AvgPoolFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          AvgPoolFunctor<T>(), input.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          output->mutable_data<T>(context.GetPlace()));
    }
  }
};

template <typename T>
struct MaxPoolGradFunctor {
  HOSTDEVICE void operator()(const T* out_grad, const size_t start,
                             const size_t end, const size_t item_dim,
                             T* in_grad, int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        if (i == *index) {
          in_grad[item_dim * i + tid] = out_grad[tid];
        } else {
          in_grad[item_dim * i + tid] = static_cast<T>(0);
        }
      }
    }
  }
};

template <typename T>
struct AvgPoolGradFunctor {
  HOSTDEVICE void operator()(const T* out_grad, const size_t start,
                             const size_t end, const size_t item_dim,
                             T* in_grad, int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        in_grad[item_dim * i + tid] = out_grad[tid] / (end - start);
      }
    }
  }
};

template <typename T, typename Range_OP>
__global__ void sequence_pool_grad_kernel(Range_OP op, const T* out_grad,
                                          const size_t* lod,
                                          const size_t lod_size,
                                          const size_t item_dim, T* in_grad,
                                          int* index) {
  int bid = blockIdx.x;
  if (bid >= lod_size) return;
  size_t start = lod[bid];
  size_t end = lod[bid + 1];
  op(&out_grad[bid * item_dim], start, end, item_dim, in_grad, index);
}

template <typename T>
class SequencePoolGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const std::string pooltype, const framework::Tensor& out_grad,
                  framework::LoDTensor* in_grad,
                  /* max pool has index */
                  const framework::Tensor* index = nullptr) {
    auto lod = in_grad->lod()[0];
    int item_dim = in_grad->numel() / out_grad.dims()[0];
    dim3 threads(1024, 1);
    dim3 block(lod.size(), 1);
    if (pooltype == "MAX") {
      sequence_pool_grad_kernel<
          T, MaxPoolGradFunctor<T>><<<block, threads, 0, context.stream()>>>(
          MaxPoolGradFunctor<T>(), out_grad.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          in_grad->mutable_data<T>(context.GetPlace()), index->data<T>());
    } else if (pooltype == "AVG") {
      sequence_pool_grad_kernel<
          T, AvgPoolGradFunctor<T>><<<block, threads, 0, context.stream()>>>(
          AvgPoolGradFunctor<T>(), out_grad.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          in_grad->mutable_data<T>(context.GetPlace()));
    }
  }
};

// sequence pooling
template class SequencePoolFunctor<platform::CUDADeviceContext, float>;
template class SequencePoolFunctor<platform::CUDADeviceContext, double>;
template class SequencePoolGradFunctor<platform::CUDADeviceContext, float>;
template class SequencePoolGradFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
