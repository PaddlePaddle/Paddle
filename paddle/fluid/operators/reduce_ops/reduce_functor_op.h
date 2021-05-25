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

namespace paddle {
namespace operators {

template <typename T>
struct CustomMin {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return (b < a) ? b : a;
  }
};

template <typename T>
struct CustomMax {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return (b > a) ? b : a;
  }
};

template <typename T>
struct CustomSum {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return b + a;
  }
};

template <typename T>
struct CustomMul {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return b * a;
  }
};

template <typename T>
struct CustomLogicalOr {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return b || a;
  }
};

template <typename T>
struct CustomLogicalAnd {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return b && a;
  }
};

}  // namespace operators
}  // namespace paddle
