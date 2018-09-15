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

#include <string>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace operators {
namespace math {

template <typename T, typename BinaryFunctor, typename UnaryFunctor>
struct BinaryCompoundFunctor {
  BinaryCompoundFunctor(const BinaryFunctor func1, const UnaryFunctor func2)
      : func1_(func1), func2_(func2) {}
  // Z = BinaryFunctor(X, UnaryFunctor(Y))

  inline HOSTDEVICE T GetOut(T x, T y) { return func1_(x, func2_(y)); }

  inline HOSTDEVICE T GetOutUseIntermediateOut(T x, T intermediat_out) {
    return func1_(x, intermediat_out);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return func2_(y); }

  BinaryFunctor func1_;
  UnaryFunctor func2_;
};

template <typename T, typename UnaryFunctor, typename BinaryFunctor>
struct UnaryCompoundFunctor {
  UnaryCompoundFunctor(const UnaryFunctor func1, const BinaryFunctor func2)
      : func1_(func1), func2_(func2) {}
  // Z = UnaryFunctor(BinaryFunctor(X, Y))

  inline HOSTDEVICE T GetOut(T x, T y) { return func1_(func2_(x, y)); }

  inline HOSTDEVICE T GetOutUseIntermediateOut(T x, T intermediat_out) {
    return func1_(intermediat_out);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return func2_(x, y); }

  UnaryFunctor func1_;
  BinaryFunctor func2_;
};

// FIXME(zcd): DBinaryFun and DUnaryFun have to method to get
// the dx, one is to use the 'out', and the other is not to use it.
// the former method will save the time of recomputing the
// 'out', but it must occupy the memory to store the 'out'.
// While the later method can avoid occupying this memory,
// but it must recompute the 'out'.
template <typename T, typename DBinaryFun, typename UnaryFun>
struct BinaryCompoundGradDxFunctor {
  BinaryCompoundGradDxFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun)
      : d_binary_fun_(d_binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    return dout * d_binary_fun_.Dx(x, unary_fun_(y));
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    return dout * d_binary_fun_.Dx(x, intermediate_out);
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
};

template <typename T, typename DBinaryFun, typename UnaryFun,
          typename DUnaryFun>
struct BinaryCompoundGradDyFunctor {
  BinaryCompoundGradDyFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun,
                              const DUnaryFun &d_unary_fun)
      : d_binary_fun_(d_binary_fun),
        unary_fun_(unary_fun),
        d_unary_fun_(d_unary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    return dout * d_binary_fun_.Dy(x, unary_fun_(y)) * d_unary_fun_(y);
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    return dout * d_binary_fun_.Dy(x, intermediate_out) *
           d_unary_fun_(y, intermediate_out);
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
  DUnaryFun d_unary_fun_;
};

template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun, bool Recomputation = true>
struct UnaryCompoundGradDxFunctor {
  UnaryCompoundGradDxFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(binary_fun_(x, y));
    } else {
      base = dout * d_unary_fun_(binary_fun_(x, y), out);
    }
    return base * d_binary_fun_.Dx(x, y);
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(intermediate_out);
    } else {
      base = dout * d_unary_fun_(intermediate_out, out);
    }
    return base * d_binary_fun_.Dx(x, y);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

template <typename T, typename DUnaryFun, typename BinaryFun,
          typename DBinaryFun, bool Recomputation = true>
struct UnaryCompoundGradDyFunctor {
  UnaryCompoundGradDyFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T operator()(T x, T y, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(binary_fun_(x, y));
    } else {
      base = dout * d_unary_fun_(binary_fun_(x, y), out);
    }
    return base * d_binary_fun_.Dy(x, y);
  }

  inline HOSTDEVICE T operator()(T x, T y, T intermediate_out, T out, T dout) {
    T base;
    if (Recomputation) {
      base = dout * d_unary_fun_(intermediate_out);
    } else {
      base = dout * d_unary_fun_(intermediate_out, out);
    }
    return base * d_binary_fun_.Dy(x, y);
  }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
