/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <type_traits>
#include "paddle/fluid/operators/jit/kernel_base.h"

namespace paddle {
namespace operators {
namespace jit {
namespace more {
namespace mkl {

template <typename T>
void VMul(const T* x, const T* y, T* z, int n);

template <typename T>
void VAdd(const T* x, const T* y, T* z, int n);

template <typename T>
void VScal(const T* a, const T* x, T* y, int n);

template <typename T>
void VExp(const T* x, T* y, int n);

template <typename T>
void VSigmoid(const T* x, T* y, int n) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(0) - y[i];
  }
  VExp(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
  }
}

template <typename T>
void VTanh(const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * x[i];
  }
  VSigmoid(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * y[i] - static_cast<T>(1);
  }
}

#define DECLARE_MKL_KERNEL(name, tuples)                             \
  template <typename T>                                              \
  class name##Kernel : public KernelMore<tuples<T>> {                \
   public:                                                           \
    name##Kernel() { this->func = name<T>; }                         \
    bool UseMe(const typename tuples<T>::attr_type&) const override; \
    const char* ImplType() const override { return "MKL"; }          \
  }

// XYZN
DECLARE_MKL_KERNEL(VMul, XYZNTuples);
DECLARE_MKL_KERNEL(VAdd, XYZNTuples);

// AXYN
DECLARE_MKL_KERNEL(VScal, AXYNTuples);

// XYN
DECLARE_MKL_KERNEL(VExp, XYNTuples);
DECLARE_MKL_KERNEL(VSigmoid, XYNTuples);
DECLARE_MKL_KERNEL(VTanh, XYNTuples);

#undef DECLARE_MKL_KERNEL

}  // namespace mkl
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle
