// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"

namespace paddle {
namespace platform {

using CUDAKernelParams = phi::backends::gpu::CUDAKernelParams;
#if CUDA_VERSION < 10010
using cudaStreamCaptureMode = phi::backends::gpu::cudaStreamCaptureMode;
#endif
using CUDAGraph = phi::backends::gpu::CUDAGraph;
using CUDAGraphCaptureModeGuard = phi::backends::gpu::CUDAGraphCaptureModeGuard;

template <typename T>
static bool IsBitwiseEqual(const T &x, const T &y) {
  return std::memcmp(&x, &y, sizeof(T)) == 0;
}

template <typename F, F f>
struct IsSameKernelHelper;

template <typename Return,
          typename... FuncArgs,
          Return (*kernel_fn)(FuncArgs...)>
struct IsSameKernelHelper<Return (*)(FuncArgs...), kernel_fn> {
 private:
  using FuncArgsTuple = decltype(std::make_tuple(std::declval<FuncArgs>()...));

  template <typename TupleT, size_t IDX, bool IsEnd /*=false*/>
  struct Impl {
    static bool Compare(const CUDAKernelParams &params, const TupleT &args) {
      using CompareT = typename std::tuple_element<IDX, FuncArgsTuple>::type;
      if (!IsBitwiseEqual<CompareT>(params.As<CompareT>(IDX),
                                    std::get<IDX>(args))) {
        return false;
      }

      constexpr auto NewIsEnd = (IDX + 1 == std::tuple_size<TupleT>::value);
      return Impl<TupleT, IDX + 1, NewIsEnd>::Compare(params, args);
    }
  };

  template <typename TupleT, size_t IDX>
  struct Impl<TupleT, IDX, true> {
    static bool Compare(const CUDAKernelParams &params, const TupleT &args) {
      return true;
    }
  };

 public:
  template <typename... Args>
  static bool Compare(const CUDAKernelParams &params, Args... args) {
    constexpr auto kNumArgs = sizeof...(FuncArgs);
    static_assert(kNumArgs == sizeof...(Args), "Argument number not match");

    auto args_tuple = std::make_tuple(args...);
    using TupleT = typename std::decay<decltype(args_tuple)>::type;
    return Impl<TupleT, 0, kNumArgs == 0>::Compare(params, args_tuple);
  }
};

}  // namespace platform
}  // namespace paddle
