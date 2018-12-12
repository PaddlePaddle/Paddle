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
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace more {
namespace mkl {

template <typename T>
void VMul(const T* x, const T* y, T* z, int n);

template <typename T>
class VMulKernel : public KernelImpl<VMulTuples<T>> {
 public:
  VMulKernel() { this->func = VMul<T>; }
  bool UseMe(int d) const override {
    if (std::is_same<T, float>::value) {
      return platform::MayIUse(platform::avx512f) && d > 512;
    } else {
      return true;
    }
  }
};

}  // namespace mkl
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle
