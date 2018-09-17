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

#include "paddle/fluid/operators/math/jit_kernel.h"
#include <string>

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

KernelPool& KernelPool::Instance() {
  static KernelPool g_jit_kernels;
  return g_jit_kernels;
}

template <>
const std::shared_ptr<LSTMKernel<float>>
KernelPool::Get<LSTMKernel<float>, int, const std::string&, const std::string&,
                const std::string&>(int d, const std::string& act_gate,
                                    const std::string& act_cand,
                                    const std::string& act_cell) {
  return nullptr;
}

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
