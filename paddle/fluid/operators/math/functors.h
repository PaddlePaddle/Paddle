/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

struct Functor {};

struct IdentityFunctor : public Functor {
  // y = x;
  template <typename T>
  inline HOSTDEVICE T operator()(T ele) const {
    return ele;
  }
};

// AddFunctor
struct AddFunctor : public Functor {
  explicit AddFunctor(Functor *fun) : func_tail1_(fun) {}
  explicit AddFunctor(Functor *fun1, Functor *fun2)
      : func_tail1_(fun1), func_tail2_(fun2) {}

  // out = f(x + y);
  template <typename T>
  inline HOSTDEVICE T operator()(T a, T b) const {
    return (*func_tail1_)(a + b);
  }

  // out = f2(f1(x, y) + z);
  template <typename T>
  inline HOSTDEVICE T operator()(T a, T b, T c) const {
    return (*func_tail2_)((*func_tail1_)(a + b), c);
  }

 private:
  std::unique_ptr<Functor> func_tail1_;
  std::unique_ptr<Functor> func_tail2_;
};

struct AddGradFunctor : public Functor {
  explicit AddGradFunctor(Functor *fun) : func_tail1_(fun) {}
  explicit AddGradFunctor(Functor *fun1, Functor *fun2)
      : func_tail1_(fun1), func_tail2_(fun2) {}

  // dx = dout * f'(x, y); dy = dout * f'(x, y);
  template <typename T>
  inline HOSTDEVICE T operator()(T a, T b, T out, T dout) const {
    return (*func_tail1_)(a, b, out, dout);
  }

  // dx = dout * f2'(f1(x, y), z) * f1'(x, y);
  // dy = dout * f2'(f1(x, y), z) * f1'(x, y);
  // dz = dout * f2'(f1(x, y), z);
  template <typename T>
  inline HOSTDEVICE T operator()(T a, T b, T c, T out, T dout) const {
    // TODO(zcd): analysis the backward.
    return (*func_tail1_)(dout);
  }

  std::unique_ptr<Functor> func_tail1_;
  std::unique_ptr<Functor> func_tail2_;
};

// ScaleFunctor
struct ScaleFunctor : public Functor {
  explicit ScaleFunctor(int64_t coeff, Functor *fun)
      : coeff_(coeff), func_tail_(fun) {}

  // out = scale * x;
  template <typename T>
  inline HOSTDEVICE T operator()(T ele) const {
    return (*func_tail_)(ele * static_cast<T>(coeff_));
  }

 private:
  int64_t coeff_;
  std::unique_ptr<Functor> func_tail_;
};

struct ScaleGradFunctor : public Functor {
  explicit ScaleGradFunctor(int64_t coeff, Functor *fun)
      : coeff_(coeff), func_tail_(fun) {}

  // out = dout * scale;
  template <typename T>
  inline HOSTDEVICE T operator()(T a) const {
    return (*func_tail_)(a)*coeff_;
  }

 private:
  int64_t coeff_;
  std::unique_ptr<Functor> func_tail_;
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
