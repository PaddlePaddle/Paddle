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
#include "paddle/pten/kernels/hybird/general/elementwise_base.h"

namespace pten {
namespace kps = paddle::operators::kernel_primitives;
enum ElementwiseType { kUnary = 1, kBinary = 2, kTernary = 3, kAny = -1 };

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

}  // namespace pten
