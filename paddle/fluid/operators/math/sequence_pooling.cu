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

#include <string>
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sequence_pooling.h"
#include "paddle/fluid/platform/cuda_primitives.h"

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
      T max_val = static_cast<T>(-FLT_MAX);
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
      output[tid] = val / static_cast<T>(end - start);
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
      output[tid] = input[item_dim * (end - 1) + tid];
    }
  }
};

template <typename T>
struct FirstPoolFunctor {
  HOSTDEVICE void operator()(const T* input, const size_t start,
                             const size_t end, const size_t item_dim, T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      output[tid] = input[item_dim * start + tid];
    }
  }
};

template <typename T, typename Range_OP>
__global__ void sequence_pool_kernel(Range_OP op, const T* input,
                                     const size_t* lod, const size_t lod_size,
                                     const size_t item_dim, T* output,
                                     int* index) {
  int bid = blockIdx.x;
  if (bid >= lod_size - 1) return;
  size_t start = lod[bid];
  size_t end = lod[bid + 1];
  int* index_offset = nullptr;
  if (index != nullptr) {
    index_offset = &index[bid * item_dim];
  }
  op(input, start, end, item_dim, &output[bid * item_dim], index_offset);
}

template <typename T>
class SequencePoolFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const std::string pooltype, const framework::LoDTensor& input,
                  framework::Tensor* output, bool is_test,
                  framework::Tensor* index = nullptr) {
    auto& lod = input.lod()[0];
    const size_t item_dim = output->numel() / output->dims()[0];
    dim3 threads(1024, 1);
    dim3 grid(lod.size(), 1);
    if (pooltype == "MAX") {
      sequence_pool_kernel<
          T, MaxPoolFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          MaxPoolFunctor<T>(), input.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          output->mutable_data<T>(context.GetPlace()), index->data<int>());
    } else if (pooltype == "AVERAGE") {
      sequence_pool_kernel<
          T, AvgPoolFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          AvgPoolFunctor<T>(), input.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          output->mutable_data<T>(context.GetPlace()), nullptr);
    } else if (pooltype == "SUM") {
      sequence_pool_kernel<
          T, SumPoolFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          SumPoolFunctor<T>(), input.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          output->mutable_data<T>(context.GetPlace()), nullptr);
    } else if (pooltype == "SQRT") {
      sequence_pool_kernel<
          T, SqrtPoolFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          SqrtPoolFunctor<T>(), input.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          output->mutable_data<T>(context.GetPlace()), nullptr);
    } else if (pooltype == "LAST") {
      sequence_pool_kernel<
          T, LastPoolFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          LastPoolFunctor<T>(), input.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          output->mutable_data<T>(context.GetPlace()), nullptr);
    } else if (pooltype == "FIRST") {
      sequence_pool_kernel<
          T, FirstPoolFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          FirstPoolFunctor<T>(), input.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          output->mutable_data<T>(context.GetPlace()), nullptr);
    } else {
      PADDLE_THROW("unsupported pooling pooltype");
    }
  }
};

template <typename T>
struct MaxPoolGradFunctor {
  HOSTDEVICE void operator()(const T* out_grad, const size_t start,
                             const size_t end, const size_t item_dim,
                             T* in_grad, const int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        if (i == index[tid]) {
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
                             T* in_grad, const int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        in_grad[item_dim * i + tid] = out_grad[tid] / (end - start);
      }
    }
  }
};

template <typename T>
struct SumPoolGradFunctor {
  HOSTDEVICE void operator()(const T* out_grad, const size_t start,
                             const size_t end, const size_t item_dim,
                             T* in_grad, const int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        in_grad[item_dim * i + tid] = out_grad[tid];
      }
    }
  }
};

template <typename T>
struct SqrtPoolGradFunctor {
  HOSTDEVICE void operator()(const T* out_grad, const size_t start,
                             const size_t end, const size_t item_dim,
                             T* in_grad, const int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        in_grad[item_dim * i + tid] =
            out_grad[tid] / (sqrt(static_cast<T>(end - start)));
      }
    }
  }
};

template <typename T>
struct LastPoolGradFunctor {
  HOSTDEVICE void operator()(const T* out_grad, const size_t start,
                             const size_t end, const size_t item_dim,
                             T* in_grad, const int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        if (i == end - 1) {
          in_grad[item_dim * i + tid] = out_grad[tid];
        } else {
          in_grad[item_dim * i + tid] = static_cast<T>(0);
        }
      }
    }
  }
};

template <typename T>
struct FirstPoolGradFunctor {
  HOSTDEVICE void operator()(const T* out_grad, const size_t start,
                             const size_t end, const size_t item_dim,
                             T* in_grad, const int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        if (i == start) {
          in_grad[item_dim * i + tid] = out_grad[tid];
        } else {
          in_grad[item_dim * i + tid] = static_cast<T>(0);
        }
      }
    }
  }
};

template <typename T, typename Range_OP>
__global__ void sequence_pool_grad_kernel(Range_OP op, const T* out_grad,
                                          const size_t* lod,
                                          const size_t lod_size,
                                          const size_t item_dim, T* in_grad,
                                          const int* index) {
  int bid = blockIdx.x;
  if (bid >= lod_size - 1) return;
  size_t start = lod[bid];
  size_t end = lod[bid + 1];
  const int* index_offset = nullptr;
  if (index != nullptr) {
    index_offset = &index[bid * item_dim];
  }
  op(&out_grad[bid * item_dim], start, end, item_dim, in_grad, index_offset);
}

template <typename T>
class SequencePoolGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const std::string pooltype, const framework::Tensor& out_grad,
                  framework::LoDTensor* in_grad,
                  /* max pool has index */
                  const framework::Tensor* index = nullptr) {
    auto& lod = in_grad->lod()[0];
    const size_t item_dim = in_grad->numel() / in_grad->dims()[0];
    dim3 threads(1024, 1);
    dim3 grid(lod.size(), 1);
    if (pooltype == "MAX") {
      sequence_pool_grad_kernel<
          T, MaxPoolGradFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          MaxPoolGradFunctor<T>(), out_grad.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          in_grad->mutable_data<T>(context.GetPlace()), index->data<int>());
    } else if (pooltype == "AVERAGE") {
      sequence_pool_grad_kernel<
          T, AvgPoolGradFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          AvgPoolGradFunctor<T>(), out_grad.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          in_grad->mutable_data<T>(context.GetPlace()), nullptr);
    } else if (pooltype == "SUM") {
      sequence_pool_grad_kernel<
          T, SumPoolGradFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          SumPoolGradFunctor<T>(), out_grad.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          in_grad->mutable_data<T>(context.GetPlace()), nullptr);
    } else if (pooltype == "SQRT") {
      sequence_pool_grad_kernel<
          T, SqrtPoolGradFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          SqrtPoolGradFunctor<T>(), out_grad.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          in_grad->mutable_data<T>(context.GetPlace()), nullptr);
    } else if (pooltype == "LAST") {
      sequence_pool_grad_kernel<
          T, LastPoolGradFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          LastPoolGradFunctor<T>(), out_grad.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          in_grad->mutable_data<T>(context.GetPlace()), nullptr);
    } else if (pooltype == "FIRST") {
      sequence_pool_grad_kernel<
          T, FirstPoolGradFunctor<T>><<<grid, threads, 0, context.stream()>>>(
          FirstPoolGradFunctor<T>(), out_grad.data<T>(),
          lod.CUDAData(context.GetPlace()), lod.size(), item_dim,
          in_grad->mutable_data<T>(context.GetPlace()), nullptr);

    } else {
      PADDLE_THROW("unsupported pooling pooltype");
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
