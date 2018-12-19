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
namespace mix {

template <typename T>
void VSigmoid(const T* x, T* y, int n);

template <typename T>
void VTanh(const T* x, T* y, int n);

#define DECLARE_MORE_KERNEL(name, tuples)                     \
  template <typename T>                                       \
  class name##Kernel : public KernelImpl<tuples<T>> {         \
   public:                                                    \
    name##Kernel() { this->func = name<T>; }                  \
    bool UseMe(typename tuples<T>::attr_type) const override; \
  }

// XYN
DECLARE_MORE_KERNEL(VSigmoid, XYNTuples);
DECLARE_MORE_KERNEL(VTanh, XYNTuples);

#undef DECLARE_MORE_KERNEL

}  // namespace mix
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle
