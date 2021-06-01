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
#include <cuda.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <cstdint>

namespace paddle {
namespace operators {

template <typename T>
struct DataBound {
  static inline T max() { return static_cast<T>(FLT_MAX); }
  static inline T min() { return static_cast<T>(-FLT_MAX); }
};

template <>
struct DataBound<float> {
  static inline float max() { return FLT_MAX; }
  static inline float min() { return -FLT_MAX; }
};

template <>
struct DataBound<double> {
  static inline double max() { return DBL_MAX; }
  static inline double min() { return -DBL_MAX; }
};

template <>
struct DataBound<int32_t> {
  static inline int32_t max() { return INT32_MAX; }
  static inline int32_t min() { return INT32_MIN; }
};

template <>
struct DataBound<int64_t> {
  static inline int64_t max() { return INT64_MAX; }
  static inline int64_t min() { return INT64_MIN; }
};

template <typename T>
struct CustomMin {
  __device__ T operator()(const T &a, const T &b) const {
    return (b < a) ? b : a;
  }
};

template <typename T>
struct CustomMax {
  __device__ T operator()(const T &a, const T &b) const {
    return (b > a) ? b : a;
  }
};

template <typename T>
struct CustomSum {
  __device__ T operator()(const T &a, const T &b) const { return b + a; }
};

template <typename T>
struct CustomMul {
  __device__ T operator()(const T &a, const T &b) const { return b * a; }
};

template <typename T>
struct CustomLogicalOr {
  __device__ T operator()(const T &a, const T &b) const { return b || a; }
};

template <typename T>
struct CustomLogicalAnd {
  __device__ T operator()(const T &a, const T &b) const { return b && a; }
};

}  // namespace operators
}  // namespace paddle
