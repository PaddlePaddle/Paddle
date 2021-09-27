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
  HOSTDEVICE explicit inline ExpFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Ty& x) const {
    return static_cast<Ty>(details::Exp(x));
  }
};

/**
 * @brief Default unary identity functor
 */
template <typename Tx, typename Ty = Tx>
struct IdentityFunctor {
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
  HOSTDEVICE explicit inline DivideFunctor(int n) : n_inv((Tx)(1.0 / n)) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x * n_inv);
  }

 private:
  Tx n_inv;
};

/**
 * @brief Default unary square functor
 */
template <typename Tx, typename Ty = Tx>
struct SquareFunctor {
  HOSTDEVICE explicit inline SquareFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x) * static_cast<Ty>(x);
  }
};

/****************************** Binary Functor ********************************/

/**
 * @brief Default binary min functor
 */
template <typename Tx, typename Ty = Tx>
struct MinFunctor {
  inline Ty initial() {
    return static_cast<Ty>(std::numeric_limits<Ty>::max());
  }

  __device__ __forceinline__ Ty operator()(const Ty& a, const Ty& b) const {
    return (b < a) ? b : a;
  }
};

/**
 * @brief Default binary max functor
 */
template <typename Tx, typename Ty = Tx>
struct MaxFunctor {
  inline Ty initial() {
    return static_cast<Ty>(std::numeric_limits<Ty>::lowest());
  }

  __device__ __forceinline__ Ty operator()(const Ty& a, const Ty& b) const {
    return (b > a) ? b : a;
  }
};

/**
 * @brief Default binary add functor
 */
template <typename Tx, typename Ty = Tx>
struct AddFunctor {
  inline Ty initial() { return static_cast<Ty>(0.0f); }

  __device__ __forceinline__ Ty operator()(const Ty& a, const Ty& b) const {
    return b + a;
  }
};

/**
 * @brief Default binary mul functor
 */
template <typename Tx, typename Ty = Tx>
struct MulFunctor {
  inline Ty initial() { return static_cast<Ty>(1.0f); }

  __device__ __forceinline__ Ty operator()(const Ty& a, const Ty& b) const {
    return b * a;
  }
};

/**
 * @brief Default binary logic or functor
 */
template <typename Tx, typename Ty = Tx>
struct LogicalOrFunctor {
  inline Ty initial() { return static_cast<Ty>(false); }

  __device__ __forceinline__ Ty operator()(const Ty& a, const Ty& b) const {
    return b || a;
  }
};

/**
 * @brief Default binary logic and functor
 */
template <typename Tx, typename Ty = Tx>
struct LogicalAndFunctor {
  inline Ty initial() { return static_cast<Ty>(true); }

  __device__ __forceinline__ Ty operator()(const Ty& a, const Ty& b) const {
    return b && a;
  }
};

/**
 * @brief Default binary sub functor
 */
template <typename Tx, typename Ty = Tx>
struct SubFunctor {
  inline Ty initial() { return static_cast<Ty>(1.0f); }

  inline HOSTDEVICE Ty operator()(const Ty& a, const Ty& b) const {
    return a - b;
  }
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
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    // For int32/int64, need to check whether the divison is zero.
    PADDLE_ENFORCE_NE(
        b, 0, platform::errors::InvalidArgument(
                  "InvalidArgumentError: Integer division by zero encountered "
                  "in (floor) divide. Please check the input value."));
    return a / b;
  }
};

/**
 * @brief Default binary floor divide functor
 */
template <typename Tx, typename Ty = Tx>
struct FloorDivFunctor {
  inline Ty initial() { return static_cast<Ty>(1.0f); }

  inline HOSTDEVICE Ty operator()(const Ty& a, const Ty& b) const {
    PADDLE_ENFORCE_NE(
        b, 0, platform::errors::InvalidArgument(
                  "InvalidArgumentError: Integer division by zero encountered "
                  "in (floor) divide. Please check the input value."));
    return static_cast<Ty>(std::trunc(a / b));
  }
};

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
