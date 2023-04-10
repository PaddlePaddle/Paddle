/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <string>

#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sequence_pooling.h"

namespace phi {
namespace funcs {

template <typename T>
struct MaxPoolFunctor {
  HOSTDEVICE void operator()(const T* input,
                             const T pad_value,
                             const size_t start,
                             const size_t end,
                             const size_t item_dim,
                             T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      T max_val = static_cast<T>(-FLT_MAX);
      int max_index = -1;
      if (start == end) {
        output[tid] = pad_value;
        index[tid] = -1;
      } else {
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
  }
};

template <typename T>
struct AvgPoolFunctor {
  HOSTDEVICE void operator()(const T* input,
                             const T pad_value,
                             const size_t start,
                             const size_t end,
                             const size_t item_dim,
                             T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      if (start == end) {
        output[tid] = pad_value;
      } else {
        T val = static_cast<T>(0);
        for (int i = start; i < end; ++i) {
          val += input[item_dim * i + tid];
        }
        // end, start is lod, so end - start != 0
        output[tid] = val / static_cast<T>(end - start);
      }
    }
  }
};

template <typename T>
struct SumPoolFunctor {
  HOSTDEVICE void operator()(const T* input,
                             const T pad_value,
                             const size_t start,
                             const size_t end,
                             const size_t item_dim,
                             T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      if (start == end) {
        output[tid] = pad_value;
      } else {
        T val = static_cast<T>(0);
        for (int i = start; i < end; ++i) {
          val += input[item_dim * i + tid];
        }
        output[tid] = val;
      }
    }
  }
};

template <typename T>
struct SqrtPoolFunctor {
  HOSTDEVICE void operator()(const T* input,
                             const T pad_value,
                             const size_t start,
                             const size_t end,
                             const size_t item_dim,
                             T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      if (start == end) {
        output[tid] = pad_value;
      } else {
        T val = static_cast<T>(0);
        for (int i = start; i < end; ++i) {
          val += input[item_dim * i + tid];
        }
        // end, start is lod, so end - start != 0
        output[tid] = val / sqrt(end - start);
      }
    }
  }
};

template <typename T>
struct LastPoolFunctor {
  HOSTDEVICE void operator()(const T* input,
                             const T pad_value,
                             const size_t start,
                             const size_t end,
                             const size_t item_dim,
                             T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      if (start == end) {
        output[tid] = pad_value;
      } else {
        output[tid] = input[item_dim * (end - 1) + tid];
      }
    }
  }
};

template <typename T>
struct FirstPoolFunctor {
  HOSTDEVICE void operator()(const T* input,
                             const T pad_value,
                             const size_t start,
                             const size_t end,
                             const size_t item_dim,
                             T* output,
                             int* index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      if (start == end) {
        output[tid] = pad_value;
      } else {
        output[tid] = input[item_dim * start + tid];
      }
    }
  }
};

template <typename T, typename Range_OP>
__global__ void sequence_pool_kernel(Range_OP op,
                                     const T* input,
                                     const T pad_value,
                                     const size_t* lod,
                                     const size_t lod_size,
                                     const size_t item_dim,
                                     T* output,
                                     int* index) {
  int bid = blockIdx.x;
  if (bid >= lod_size - 1) return;
  size_t start = lod[bid];
  size_t end = lod[bid + 1];
  int* index_offset = nullptr;
  if (index != nullptr) {
    index_offset = &index[bid * item_dim];
  }
  op(input,
     pad_value,
     start,
     end,
     item_dim,
     &output[bid * item_dim],
     index_offset);
}

template <typename T>
class SequencePoolFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const std::string pooltype,
                  T pad_value,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* output,
                  bool is_test,
                  phi::DenseTensor* index = nullptr) {
    auto lod_level = input.lod().size();
    auto& lod = input.lod()[lod_level - 1];
    const size_t item_dim = output->numel() / output->dims()[0];
    dim3 threads(1024, 1);
    dim3 grid(std::max(static_cast<int>(lod.size()) - 1, 1), 1);
    phi::MixVector<size_t> mix_vector(&lod);
    if (pooltype == "MAX") {
      sequence_pool_kernel<T, MaxPoolFunctor<T>>
          <<<grid, threads, 0, context.stream()>>>(
              MaxPoolFunctor<T>(),
              input.data<T>(),
              pad_value,
              mix_vector.CUDAData(context.GetPlace()),
              lod.size(),
              item_dim,
              output->mutable_data<T>(context.GetPlace()),
              index->data<int>());
    } else if (pooltype == "AVERAGE") {
      sequence_pool_kernel<T, AvgPoolFunctor<T>>
          <<<grid, threads, 0, context.stream()>>>(
              AvgPoolFunctor<T>(),
              input.data<T>(),
              pad_value,
              mix_vector.CUDAData(context.GetPlace()),
              lod.size(),
              item_dim,
              output->mutable_data<T>(context.GetPlace()),
              nullptr);
    } else if (pooltype == "SUM") {
      sequence_pool_kernel<T, SumPoolFunctor<T>>
          <<<grid, threads, 0, context.stream()>>>(
              SumPoolFunctor<T>(),
              input.data<T>(),
              pad_value,
              mix_vector.CUDAData(context.GetPlace()),
              lod.size(),
              item_dim,
              output->mutable_data<T>(context.GetPlace()),
              nullptr);
    } else if (pooltype == "SQRT") {
      sequence_pool_kernel<T, SqrtPoolFunctor<T>>
          <<<grid, threads, 0, context.stream()>>>(
              SqrtPoolFunctor<T>(),
              input.data<T>(),
              pad_value,
              mix_vector.CUDAData(context.GetPlace()),
              lod.size(),
              item_dim,
              output->mutable_data<T>(context.GetPlace()),
              nullptr);
    } else if (pooltype == "LAST") {
      sequence_pool_kernel<T, LastPoolFunctor<T>>
          <<<grid, threads, 0, context.stream()>>>(
              LastPoolFunctor<T>(),
              input.data<T>(),
              pad_value,
              mix_vector.CUDAData(context.GetPlace()),
              lod.size(),
              item_dim,
              output->mutable_data<T>(context.GetPlace()),
              nullptr);
    } else if (pooltype == "FIRST") {
      sequence_pool_kernel<T, FirstPoolFunctor<T>>
          <<<grid, threads, 0, context.stream()>>>(
              FirstPoolFunctor<T>(),
              input.data<T>(),
              pad_value,
              mix_vector.CUDAData(context.GetPlace()),
              lod.size(),
              item_dim,
              output->mutable_data<T>(context.GetPlace()),
              nullptr);
    } else {
      PADDLE_THROW(errors::InvalidArgument(
          "unsupported pooling pooltype: %s. Only support \"MAX\", "
          "\"AVERAGE\", \"SUM\", \"SQRT\", \"LAST\" and \"FIRST\"",
          pooltype));
    }
  }
};

// sequence pooling
template class SequencePoolFunctor<phi::GPUContext, float>;
template class SequencePoolFunctor<phi::GPUContext, double>;

}  // namespace funcs
}  // namespace phi
