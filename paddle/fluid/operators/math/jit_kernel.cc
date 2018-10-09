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
#include <iostream>
#include <string>

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

namespace jit = platform::jit;

KernelPool& KernelPool::Instance() {
  static thread_local KernelPool g_jit_kernels;
  return g_jit_kernels;
}

std::shared_ptr<const Kernel> KernelPool::Get(const std::string& key) const {
  if (kers_.find(key) == kers_.end()) {
    return nullptr;
  }
  return kers_.at(key);
}

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
