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
#include "paddle/common/macros.h"
namespace phi {
namespace funcs {

// Z = BinaryFunctor(X, UnaryFunctor(Y))
template <typename T, typename BinaryFunctor, typename UnaryFunctor>
struct BinaryCompoundFunctor {
  BinaryCompoundFunctor(const BinaryFunctor func1, const UnaryFunctor func2)
      : func1_(func1), func2_(func2) {}

  inline HOSTDEVICE T GetOut(T x, T y) { return func1_(x, func2_(y)); }

  inline HOSTDEVICE T GetOutUseIntermediateOut(T x, T intermediate_out) {
    return func1_(x, intermediate_out);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x UNUSED, T y) { return func2_(y); }

  BinaryFunctor func1_;
  UnaryFunctor func2_;
};

// Z = UnaryFunctor(BinaryFunctor(X, Y))
template <typename T, typename UnaryFunctor, typename BinaryFunctor>
struct UnaryCompoundFunctor {
  UnaryCompoundFunctor(const UnaryFunctor func1, const BinaryFunctor func2)
      : func1_(func1), func2_(func2) {}

  inline HOSTDEVICE T GetOut(T x, T y) { return func1_(func2_(x, y)); }

  inline HOSTDEVICE T GetOutUseIntermediateOut(T x UNUSED, T intermediate_out) {
    return func1_(intermediate_out);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return func2_(x, y); }

  UnaryFunctor func1_;
  BinaryFunctor func2_;
};

// Z = BinaryFunctor(X, UnaryFunctor(Y))
template <typename T, typename DBinaryFun, typename UnaryFun>
struct BinaryCompoundGradDxFunctor {
  BinaryCompoundGradDxFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun)
      : d_binary_fun_(d_binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T Recompute(T x, T y, T out UNUSED, T dout) {
    return dout * d_binary_fun_.Dx(x, unary_fun_(y));
  }

  inline HOSTDEVICE T UseIntermediateOut(
      T x, T y UNUSED, T intermediate_out, T out UNUSED, T dout) {
    return dout * d_binary_fun_.Dx(x, intermediate_out);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x UNUSED, T y) {
    return unary_fun_(y);
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
};

// Z = BinaryFunctor(X, UnaryFunctor(Y))
template <typename T,
          typename DBinaryFun,
          typename UnaryFun,
          typename DUnaryFun,
          bool InPlace>
struct BinaryCompoundGradDyFunctor {
  BinaryCompoundGradDyFunctor(const DBinaryFun &d_binary_fun,
                              const UnaryFun &unary_fun,
                              const DUnaryFun &d_unary_fun)
      : d_binary_fun_(d_binary_fun),
        unary_fun_(unary_fun),
        d_unary_fun_(d_unary_fun) {}

  inline HOSTDEVICE T Recompute(T x, T y, T out UNUSED, T dout) {
    return dout * d_binary_fun_.Dy(x, unary_fun_(y)) * d_unary_fun_.UseX(y);
  }

  inline HOSTDEVICE T
  UseIntermediateOut(T x, T y, T intermediate_out, T out UNUSED, T dout) {
    if (InPlace) {
      return dout * d_binary_fun_.Dy(x, intermediate_out) *
             d_unary_fun_.UseOut(intermediate_out);
    } else {
      return dout * d_binary_fun_.Dy(x, intermediate_out) *
             d_unary_fun_.UseXAndOut(y, intermediate_out);
    }
  }

  inline HOSTDEVICE T GetIntermediateOut(T x UNUSED, T y) {
    return unary_fun_(y);
  }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
  DUnaryFun d_unary_fun_;
};

// Z = UnaryFunctor(BinaryFunctor(X, Y))
template <typename T,
          typename DUnaryFun,
          typename BinaryFun,
          typename DBinaryFun,
          bool InPlace>
struct UnaryCompoundGradDxFunctor {
  UnaryCompoundGradDxFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T Recompute(T x, T y, T out, T dout) {
    T base;
    if (InPlace) {
      base = dout * d_unary_fun_.UseOut(out);
    } else {
      base = dout * d_unary_fun_.UseXAndOut(binary_fun_(x, y), out);
    }
    return base * d_binary_fun_.Dx(x, y);
  }

  inline HOSTDEVICE T
  UseIntermediateOut(T x, T y, T intermediate_out, T out, T dout) {
    T base;
    if (InPlace) {
      base = dout * d_unary_fun_.UseOut(out);
    } else {
      base = dout * d_unary_fun_.UseXAndOut(intermediate_out, out);
    }
    return base * d_binary_fun_.Dx(x, y);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return binary_fun_(x, y); }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

// Z = UnaryFunctor(BinaryFunctor(X, Y))
template <typename T,
          typename DUnaryFun,
          typename BinaryFun,
          typename DBinaryFun,
          bool InPlace>
struct UnaryCompoundGradDyFunctor {
  UnaryCompoundGradDyFunctor(const DUnaryFun &d_unary_fun,
                             const BinaryFun &binary_fun,
                             const DBinaryFun &d_binary_fun)
      : d_unary_fun_(d_unary_fun),
        binary_fun_(binary_fun),
        d_binary_fun_(d_binary_fun) {}

  inline HOSTDEVICE T Recompute(T x, T y, T out, T dout) {
    T base;
    if (InPlace) {
      base = dout * d_unary_fun_.UseOut(out);
    } else {
      base = dout * d_unary_fun_.UseXAndOut(binary_fun_(x, y), out);
    }
    return base * d_binary_fun_.Dy(x, y);
  }

  inline HOSTDEVICE T
  UseIntermediateOut(T x, T y, T intermediate_out, T out, T dout) {
    T base;
    if (InPlace) {
      base = dout * d_unary_fun_.UseOut(out);
    } else {
      base = dout * d_unary_fun_.UseXAndOut(intermediate_out, out);
    }
    return base * d_binary_fun_.Dy(x, y);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return binary_fun_(x, y); }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
  DBinaryFun d_binary_fun_;
};

// Z = BinaryFunctor(X, UnaryFunctor(Y))
template <typename T, typename DBinaryFun, typename UnaryFun>
struct BinaryCompoundGradDIntermediateOutFunctor {
  BinaryCompoundGradDIntermediateOutFunctor(const DBinaryFun &d_binary_fun,
                                            const UnaryFun &unary_fun)
      : d_binary_fun_(d_binary_fun), unary_fun_(unary_fun) {}

  inline HOSTDEVICE T Recompute(T x, T y, T out UNUSED, T dout) {
    return dout * d_binary_fun_.Dy(x, unary_fun_(y));
  }

  inline HOSTDEVICE T UseIntermediateOut(T x,
                                         T intermediate_out,
                                         T out UNUSED,
                                         T dout) {
    return dout * d_binary_fun_.Dy(x, intermediate_out);
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return unary_fun_(y); }

 private:
  DBinaryFun d_binary_fun_;
  UnaryFun unary_fun_;
};

// Z = UnaryFunctor(BinaryFunctor(X, Y))
template <typename T, typename DUnaryFun, typename BinaryFun, bool InPlace>
struct UnaryCompoundGradDIntermediateFunctor {
  UnaryCompoundGradDIntermediateFunctor(const DUnaryFun &d_unary_fun,
                                        const BinaryFun &binary_fun)
      : d_unary_fun_(d_unary_fun), binary_fun_(binary_fun) {}

  inline HOSTDEVICE T Recompute(T x, T y, T out, T dout) {
    if (InPlace) {
      return dout * d_unary_fun_.UseOut(out);
    } else {
      return dout * d_unary_fun_.UseXAndOut(binary_fun_(x, y), out);
    }
  }

  inline HOSTDEVICE T UseIntermediateOut(T x UNUSED,
                                         T intermediate_out,
                                         T out,
                                         T dout) {
    if (InPlace) {
      return dout * d_unary_fun_.UseOut(out);
    } else {
      return dout * d_unary_fun_.UseXAndOut(intermediate_out, out);
    }
  }

  inline HOSTDEVICE T GetIntermediateOut(T x, T y) { return binary_fun_(x, y); }

 private:
  DUnaryFun d_unary_fun_;
  BinaryFun binary_fun_;
};

}  // namespace funcs
}  // namespace phi
