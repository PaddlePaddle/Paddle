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
#include <math.h>
#include <limits>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

namespace paddle {
namespace operators {

// Post processing function for sum, max, min, prod, any
template <typename Tx, typename Ty = Tx>
struct IdentityFunctor {
  __device__ explicit inline IdentityFunctor() {}

  __device__ explicit inline IdentityFunctor(int n) {}

  __device__ inline Ty operator()(const Tx &x) const {
    return static_cast<Ty>(x);
  }
};

// Post processing function for mean
template <typename T>
struct DivideFunctor {
  __device__ explicit inline DivideFunctor(int n) : n_inv((T)(1.0 / n)) {}

  __device__ inline T operator()(const T &x) const { return x * n_inv; }

 private:
  T n_inv;
};

template <typename Tx, typename Ty = Tx>
struct CustomMin {
  using Transformer = IdentityFunctor<Tx>;

  HOSTDEVICE __forceinline__ Ty initial() {
    return std::numeric_limits<Ty>::max();
  }

  __device__ __forceinline__ Ty operator()(const Ty &a, const Ty &b) const {
    return (b < a) ? b : a;
  }
};

template <typename Tx, typename Ty = Tx>
struct CustomMax {
  using Transformer = IdentityFunctor<Tx>;

  HOSTDEVICE __forceinline__ Ty initial() {
    return std::numeric_limits<Ty>::min();
  }

  __device__ __forceinline__ Ty operator()(const Ty &a, const Ty &b) const {
    return (b > a) ? b : a;
  }
};

// for cub::Reduce
template <typename Tx, typename Ty = Tx>
struct CustomSum {
  using Transformer = IdentityFunctor<Tx, Ty>;

  HOSTDEVICE __forceinline__ Ty initial() { return static_cast<Ty>(0.0f); }

  __device__ __forceinline__ Ty operator()(const Ty &a, const Ty &b) const {
    return b + a;
  }
};

template <typename Tx, typename Ty = Tx>
struct CustomMean {
  using Transformer = DivideFunctor<Tx>;

  HOSTDEVICE __forceinline__ Ty initial() { return static_cast<Ty>(0.0f); }

  __device__ __forceinline__ Ty operator()(const Ty &a, const Ty &b) const {
    return b + a;
  }
};

template <typename Tx, typename Ty = Tx>
struct CustomMul {
  using Transformer = IdentityFunctor<Tx>;

  HOSTDEVICE __forceinline__ Ty initial() { return static_cast<Ty>(1.0f); }

  __device__ __forceinline__ Ty operator()(const Ty &a, const Ty &b) const {
    return b * a;
  }
};

template <typename Tx, typename Ty = Tx>
struct CustomLogicalOr {
  using Transformer = IdentityFunctor<Tx>;

  HOSTDEVICE __forceinline__ Ty initial() { return static_cast<Ty>(false); }

  __device__ __forceinline__ Ty operator()(const Ty &a, const Ty &b) const {
    return b || a;
  }
};

template <typename Tx, typename Ty = Tx>
struct CustomLogicalAnd {
  using Transformer = IdentityFunctor<Tx>;

  HOSTDEVICE __forceinline__ Ty initial() { return static_cast<Ty>(true); }

  __device__ __forceinline__ Ty operator()(const Ty &a, const Ty &b) const {
    return b && a;
  }
};

}  // namespace operators
}  // namespace paddle
