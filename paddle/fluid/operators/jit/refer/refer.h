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
#include "paddle/fluid/operators/jit/helper.h"
#include "paddle/fluid/operators/jit/kernel_base.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace jit {
namespace refer {

// Refer code only focus on correctness
template <typename T>
void VMul(const T* x, const T* y, T* z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}

template <typename T>
void VAdd(const T* x, const T* y, T* z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

template <typename T>
void VAddRelu(const T* x, const T* y, T* z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
    z[i] = z[i] > 0 ? z[i] : 0;
  }
}

template <typename T>
void VSub(const T* x, const T* y, T* z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] - y[i];
  }
}

template <typename T>
void VScal(const T* a, const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = a[0] * x[i];
  }
}

template <typename T>
void VAddBias(const T* a, const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = a[0] + x[i];
  }
}

#define DECLARE_REFER_KERNEL(name, tuples)             \
  template <typename T>                                \
  class name##Kernel : public ReferKernel<tuples<T>> { \
   public:                                             \
    name##Kernel() { this->func = name<T>; }           \
  }

// const T* x, const T* y, T* z, int n
DECLARE_REFER_KERNEL(VMul, XYZNTuples);
DECLARE_REFER_KERNEL(VAdd, XYZNTuples);
DECLARE_REFER_KERNEL(VAddRelu, XYZNTuples);
DECLARE_REFER_KERNEL(VSub, XYZNTuples);

// const T* a, const T* x, T* y, int n
DECLARE_REFER_KERNEL(VScal, AXYNTuples);
DECLARE_REFER_KERNEL(VAddBias, AXYNTuples);

#undef DECLARE_REFER_KERNEL

}  // namespace refer
}  // namespace jit
}  // namespace operators
}  // namespace paddle
