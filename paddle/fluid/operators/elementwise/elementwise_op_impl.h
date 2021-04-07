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
#pragma once

#include <algorithm>
#include <utility>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#ifdef __NVCC__
#include <cuda.h>
#include <cuda_fp16.h>
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#endif

namespace paddle {
namespace operators {

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
};

template <typename T>
inline int VectorizedSize_V2(const T *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2 = std::alignment_of<AlignedVector<T, 2>>::value;  // NOLINT
  constexpr int vec4 = std::alignment_of<AlignedVector<T, 4>>::value;  // NOLINT
  constexpr int vec8 = std::alignment_of<AlignedVector<T, 8>>::value;  // NOLINT
  if (address % vec8 == 0) {
    return 8;
  } else if (address % vec4 == 0) {
    return 4;
  } else if (address % vec2 == 0) {
    return 2;
  }
  return 1;
}

template <typename T>
struct SameDimsData {
  int data_num = 0;
  T *out = nullptr;
  const T *in0 = nullptr;
  const T *in1 = nullptr;
  SameDimsData(int data_num, T *out, const T *in0, const T *in1 = nullptr)
      : data_num(data_num), out(out), in0(in0), in1(in1) {}

  int GetVectorizedSize() {
    int vec_size = 1;
    vec_size = std::min<int>(vec_size, VectorizedSize_V2<T>(out));
    vec_size = std::min<int>(vec_size, VectorizedSize_V2<T>(in0));
    if (in1 != nullptr) {
      vec_size = std::min<int>(vec_size, VectorizedSize_V2<T>(in1));
    }
    return vec_size;
  }
};

template <typename T, typename Functor>
void same_dims_launch_kernel(const framework::ExecutionContext &ctx,
                             SameDimsData<T> data, int64_t size, Functor func) {
}
}  // namespace operators
}  // namespace paddle
