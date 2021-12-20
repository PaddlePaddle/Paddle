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

#include "paddle/fluid/platform/eigen_ext.h"

namespace paddle {
namespace operators {
namespace kernel_primitives {
namespace details {

static __device__ __forceinline__ platform::float16 Exp(platform::float16 x) {
  return ::Eigen::numext::exp(x);
}

static __device__ __forceinline__ float Exp(float x) { return expf(x); }

static __device__ __forceinline__ double Exp(double x) { return exp(x); }

static __device__ __forceinline__ platform::float16 Log(platform::float16 x) {
  return ::Eigen::numext::log(x);
}

static __device__ __forceinline__ float Log(float x) { return logf(x); }

static __device__ __forceinline__ double Log(double x) { return log(x); }

}  // namespace details

/******************************** Unary Functor *******************************/

/**
 * @brief Default unary exp functor
 */
template <typename Tx, typename Ty = Tx>
struct ExpFunctor {
  HOSTDEVICE inline ExpFunctor() {}

  HOSTDEVICE explicit inline ExpFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(details::Exp(x));
  }
};

/**
 * @brief Default unary identity functor
 */
template <typename Tx, typename Ty = Tx>
struct IdentityFunctor {
  HOSTDEVICE inline IdentityFunctor() {}

  HOSTDEVICE explicit inline IdentityFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x);
  }
};

/**
 * @brief Default unary div functor. Divide by a constant
 */
template <typename Tx, typename Ty = Tx>
struct DivideFunctor {
  HOSTDEVICE inline DivideFunctor() { n_inv = static_cast<Tx>(1.0f); }

  HOSTDEVICE explicit inline DivideFunctor(int n) : n_inv((Tx)(1.0 / n)) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x * n_inv);
  }

 private:
  Tx n_inv;
};

/**
 * @brief Default inverse functor
 */
template <typename Tx, typename Ty = Tx>
struct InverseFunctor {
  HOSTDEVICE inline InverseFunctor() {}

  HOSTDEVICE explicit inline InverseFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(-x);
  }
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

  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
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

  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return (b > a) ? b : a;
  }
};

/**
 * @brief Default binary add functor
 */
template <typename T>
struct AddFunctor {
  inline T initial() { return static_cast<T>(0.0f); }

  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return b + a;
  }
};

/**
 * @brief Default binary add functor
 */
template <typename T>
struct MulFunctor {
  inline T initial() { return static_cast<T>(1.0f); }

  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return b * a;
  }
};

/**
 * @brief Default binary logic or functor
 */
template <typename T>
struct LogicalOrFunctor {
  inline T initial() { return static_cast<T>(false); }

  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return b || a;
  }
};

/**
 * @brief Default binary logic and functor
 */
template <typename T>
struct LogicalAndFunctor {
  inline T initial() { return static_cast<T>(true); }

  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return b && a;
  }
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
    // For int32/int64, need to check whether the divison is zero.
    PADDLE_ENFORCE_NE(b, 0,
                      platform::errors::InvalidArgument(
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
    PADDLE_ENFORCE_NE(b, 0,
                      platform::errors::InvalidArgument(
                          "Integer division by zero encountered "
                          "in (floor) divide. Please check the input value."));
    return static_cast<T>(std::trunc(a / b));
  }
};

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
