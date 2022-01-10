/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/function_traits.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/funcs/elementwise_base.h"

namespace pten {
namespace kps = paddle::operators::kernel_primitives;
enum ElementwiseType { kUnary = 1, kBinary = 2, kTernary = 3, kAny = -1 };

/* Packing scalar type T(float, int etc.) into Array<T, NumOuts> type
   for supporting multiple-output feature in elementwise system.*/
template <class T, int Num>
using ConditionalT =
    typename std::conditional_t<Num == 1, T, paddle::framework::Array<T, Num>>;

template <typename InT,
          typename OutT,
          int VecSize,
          typename Functor,
          int Arity,
          bool CallElementwiseAny = false>
struct ElementwisePrimitiveCaller {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result);
};

namespace detail {
template <class F, class Tuple, std::size_t... INDEX>
// GCC/Clang need the decltype() return type
constexpr decltype(auto) apply_impl(F &&f,
                                    Tuple &&t,
                                    std::index_sequence<INDEX...>) {
  return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}
}  // namespace detail

template <class F, class Tuple>
constexpr decltype(auto) apply(F &&f, Tuple &&t) {
  return detail::apply_impl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <typename InT,
          typename OutT,
          int VecSize,
          typename Functor,
          typename ArgsTuple,
          int Arity>
struct SameDimsElementwisePrimitiveCaller {
  __device__ inline void operator()(Functor func,
                                    ArgsTuple *args,
                                    OutT *result) {
#pragma unroll
    for (int idx = 0; idx < VecSize; ++idx) {
      result[idx] = static_cast<OutT>(apply(func, args[idx]));
    }
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor, int Arity>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, Arity, true> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseAny<InT, OutT, VecSize, 1, 1, Arity, Functor>(
        result, args, func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 1, false> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseUnary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 2, false> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], args[1], func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 3, false> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseTernary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], args[1], args[2], func);
  }
};

template <typename OutT, int VecSize, bool IsBoundary, int NumOuts>
struct ElementwiseWriteDataCaller {
  __device__ __forceinline__ void operator()(
      paddle::framework::Array<OutT *, NumOuts> outs,
      ConditionalT<OutT, NumOuts> src[VecSize],
      int block_offset,
      int num) {
    OutT dst[NumOuts][VecSize];
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
#pragma unroll
      for (int j = 0; j < NumOuts; ++j) {
        dst[j][i] = (src[i])[j];
      }
    }
#pragma unroll
    for (int i = 0; i < NumOuts; ++i) {
      kps::WriteData<OutT, VecSize, 1, 1, IsBoundary>(
          outs[i] + block_offset, dst[i], num);
    }
  }
};

template <typename OutT, int VecSize, bool IsBoundary>
struct ElementwiseWriteDataCaller<OutT, VecSize, IsBoundary, 1> {
  __device__ __forceinline__ void operator()(
      paddle::framework::Array<OutT *, 1> outs,
      OutT src[VecSize],
      int block_offset,
      int num) {
    kps::WriteData<OutT, VecSize, 1, 1, IsBoundary>(
        outs[0] + block_offset, src, num);
  }
};

}  // namespace pten
