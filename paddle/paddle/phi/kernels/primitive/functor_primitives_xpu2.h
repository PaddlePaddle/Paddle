// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/debug.h"
#include "xpu/kernel/math.h"

namespace phi {
namespace kps {
/**
 * @brief Default unary identity functor
 */
template <typename Tx, typename Ty = Tx>
struct IdentityFunctor {
#ifdef PADDLE_WITH_XPU_KP
  HOSTDEVICE inline IdentityFunctor() {}
  HOSTDEVICE explicit inline IdentityFunctor(int n) {}
  HOSTDEVICE Ty operator()(const Tx x) const { return static_cast<Ty>(x); }
  HOSTDEVICE inline void SetDiv(int n) {}
#else
  inline IdentityFunctor() {}

  explicit inline IdentityFunctor(int n) {}

  inline Ty operator()(const Tx& x) const { return static_cast<Ty>(x); }
  __device__ inline IdentityFunctor() {}

  __device__ explicit inline IdentityFunctor(int n) {}

  __device__ inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x);
  }
  __device__ inline void SetDiv(int n) {}
#endif
};

/**
 * @brief Default unary div functor. Divide by a constant
 */
template <typename Tx, typename Ty = Tx>
struct DivideFunctor {
  inline DivideFunctor() { n_inv = static_cast<Tx>(1.0f); }

  explicit inline DivideFunctor(int n)
      : n_inv(static_cast<Tx>(1.0f / (static_cast<float>(n)))) {}

  inline Ty operator()(const Tx& x) const { return static_cast<Ty>(x * n_inv); }

  __device__ inline DivideFunctor() { n_inv = static_cast<Tx>(1.0f); }

  __device__ inline DivideFunctor(int n)
      : n_inv(static_cast<Tx>(1.0f / (static_cast<float>(n)))) {}

  __device__ inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x * n_inv);
  }

  __device__ inline void SetDiv(int n) {
    n_inv = static_cast<Tx>(1.0f / (static_cast<float>(n)));
  }

 private:
  Tx n_inv;
};

/**
 * @brief Default unary square functor
 */
template <typename Tx, typename Ty = Tx>
struct SquareFunctor {
  HOSTDEVICE inline SquareFunctor() {}

  HOSTDEVICE explicit inline SquareFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x) * static_cast<Ty>(x);
  }
};

/****************************** Binary Functor ********************************/

/**
 * @brief Default binary min functor
 */
template <typename T>
struct MinFunctor {
  inline T initial() { return static_cast<T>(std::numeric_limits<T>::max()); }

  __device__ T operator()(const T& a, const T& b) const {
    return (b < a) ? b : a;
  }
};

/**
 * @brief Default binary max functor
 */
template <typename T>
struct MaxFunctor {
  inline T initial() {
    return static_cast<T>(std::numeric_limits<T>::lowest());
  }

  __device__ T operator()(const T& a, const T& b) const {
    return (b > a) ? b : a;
  }
};

/**
 * @brief Default binary add functor
 */
template <typename T>
struct AddFunctor {
  inline T initial() { return static_cast<T>(0.0f); }

  __device__ T operator()(const T a, const T b) const { return b + a; }
};

/**
 * @brief Default binary add functor
 */
template <typename T>
struct MulFunctor {
  inline T initial() { return static_cast<T>(1.0f); }

  __device__ T operator()(const T& a, const T& b) const { return b * a; }
};

/**
 * @brief Default binary logic or functor
 */
template <typename T>
struct LogicalOrFunctor {
  inline T initial() { return static_cast<T>(false); }

  __device__ T operator()(const T& a, const T& b) const { return b || a; }
};

/**
 * @brief Default binary logic and functor
 */
template <typename T>
struct LogicalAndFunctor {
  inline T initial() { return static_cast<T>(true); }

  __device__ T operator()(const T& a, const T& b) const { return b && a; }
};

/**
 * @brief Default binary sub functor
 */
template <typename T>
struct SubFunctor {
  inline T initial() { return static_cast<T>(0.0f); }

  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a - b; }
};

/**
 * @brief Default binary div functor
 */
template <typename T, typename Enable = void>
struct DivFunctor {
  inline T initial() { return static_cast<T>(1.0f); }

  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a / b; }
};

template <typename T>
struct DivFunctor<T,
                  typename std::enable_if<std::is_integral<T>::value>::type> {
  inline T initial() { return static_cast<T>(1.0f); }

  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    // For int32/int64, need to check whether the division is zero.
    PADDLE_ENFORCE_NE(b,
                      0,
                      phi::errors::InvalidArgument(
                          "Integer division by zero encountered "
                          "in (floor) divide. Please check the input value."));
    return a / b;
  }
};

/**
 * @brief Default binary floor divide functor
 */
template <typename T>
struct FloorDivFunctor {
  inline T initial() { return static_cast<T>(1.0f); }

  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    PADDLE_ENFORCE_NE(b,
                      0,
                      phi::errors::InvalidArgument(
                          "Integer division by zero encountered "
                          "in (floor) divide. Please check the input value."));
    return static_cast<T>(std::trunc(a / b));
  }
};

}  // namespace kps
}  // namespace phi
