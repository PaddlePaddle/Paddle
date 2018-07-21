/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

// AddFunctor
template <typename T>
struct AddFunctor {
  // out = x + y;
  inline HOSTDEVICE T operator()(const T x, const T y) const { return x + y; }
};

template <typename T>
struct AddGradFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y) const { return 1; }
};

// ScaleFunctor
template <typename T>
struct ScaleFunctor {
  explicit ScaleFunctor(const T coeff) : coeff_(coeff) {}

  inline HOSTDEVICE T operator()(const T ele) const { return ele * coeff_; }

 private:
  T coeff_;
};

template <typename T>
struct ScaleGradFunctor {
  explicit ScaleGradFunctor(T coeff) : coeff_(coeff) {}

  inline HOSTDEVICE T operator()(const T x) const { return coeff_; }

 private:
  T coeff_;
};

// ReluFunctor
template <typename T>
struct ReluFunctor {
  inline HOSTDEVICE T operator()(const T x) const { return x * (x > 0); }
};

template <typename T>
struct ReluGradFunctor {
  inline HOSTDEVICE T operator()(const T x) const { return x > 0 ? 1 : 0; }
};

// for example: z = x + scale * y
// fun1: t = scale * y
// fun2: z = x + t
template <typename T, typename BinaryFun, typename UnaryFun>
struct BinaryCompoundFunctor {
  BinaryCompoundFunctor(const BinaryFun &binary_fun, const UnaryFun &unary_fun)
      : binary_fun_(binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y) const {
    return binary_fun_(x, unary_fun_(y));
  }

 private:
  BinaryFun binary_fun_;
  UnaryFun unary_fun_;
};

// for example: z = x + scale * y
// fun1: z = (x + t)'
// fun2: t = scale * y
template <typename T, typename DBinaryFun, typename UnaryFun>
struct BinaryCompoundGradDxFunctor {
  BinaryCompoundGradDxFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun)
      : d_binary_fun_(d_binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y, const T out,
                                 const T dout) const {
    return dout * d_binary_fun_(x, unary_fun_(y));
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
};

// for example: z = x + scale * y
// fun1: z = (x + t)'
// fun2: t = scale * y
// fun3: t = (scale * x)'
template <typename T, typename DBinaryFun, typename UnaryFun,
          typename DUnaryFun>
struct BinaryCompoundGradDyFunctor {
  BinaryCompoundGradDyFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun,
                              const DUnaryFun &d_unary_fun)
      : d_binary_fun_(d_binary_fun),
        unary_fun_(unary_fun),
        d_unary_fun_(d_unary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y, const T out,
                                 const T dout) const {
    return dout * d_binary_fun_(unary_fun_(y), x) * d_unary_fun_(y);
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
  DUnaryFun d_unary_fun_;
};

// for example: z = scale * (x + y)
// fun1: t = scale * x
// fun2: t = x + y
template <typename T, typename UnaryFun, typename BinaryFun>
struct UnaryCompoundFunctor {
  UnaryCompoundFunctor(const UnaryFun &unary_fun, const BinaryFun &binary_fun)
      : unary_fun_(unary_fun), binary_fun_(binary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y) const {
    return unary_fun_(binary_fun_(x, y));
  }

 private:
  UnaryFun unary_fun_;
  BinaryFun binary_fun_;
};

// for example: z = scale * (x + y)
// fun1: t = (scale * x)'
// fun2: t = x + y
// fun3: t = (x + y)'
template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun>
struct UnaryCompoundGradDxFunctor {
  UnaryCompoundGradDxFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y, const T out,
                                 const T dout) const {
    auto base = dout * d_unary_fun_(binary_fun_(x, y));
    return base * d_binary_fun_(x, y);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

// for example: z = scale * (x + y)
// fun1: t = (scale * x)'
// fun2: t = x + y
// fun3: t = (x + y)'
template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun>
struct UnaryCompoundGradDyFunctor {
  UnaryCompoundGradDyFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(const T x, const T y, const T out,
                                 const T dout) const {
    auto base = dout * d_unary_fun_(binary_fun_(x, y));
    return base * d_binary_fun_(y, x);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
