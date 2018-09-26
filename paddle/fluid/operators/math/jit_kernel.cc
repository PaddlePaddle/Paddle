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

namespace jit = platform::jit;

KernelPool& KernelPool::Instance() {
  static KernelPool g_jit_kernels;
  return g_jit_kernels;
}

template <>
const std::shared_ptr<VMulKernel<float>> KernelPool::Get<VMulKernel<float>>(
    int d) {
  std::string key = "f" + std::to_string(d);
  if (kers_.find(key) == kers_.end()) {
    auto p = std::make_shared<VMulKernel<float>>(d);
    kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});
    return p;
  }
  return std::dynamic_pointer_cast<VMulKernel<float>>(kers_.at(key));
}

template <>
const std::shared_ptr<VMulKernel<double>> KernelPool::Get<VMulKernel<double>>(
    int d) {
  std::string key = "d" + std::to_string(d);
  if (kers_.find(key) == kers_.end()) {
    auto p = std::make_shared<VMulKernel<double>>(d);
    kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});
    return p;
  }
  return std::dynamic_pointer_cast<VMulKernel<double>>(kers_.at(key));
}

template <>
const std::shared_ptr<LSTMKernel<float>>
KernelPool::Get<LSTMKernel<float>, int, const std::string&, const std::string&,
                const std::string&>(int d, const std::string& act_gate,
                                    const std::string& act_cand,
                                    const std::string& act_cell) {
  std::string key = "f" + std::to_string(d) + act_gate + act_cand + act_cell;
  if (kers_.find(key) == kers_.end()) {
    auto p =
        std::make_shared<LSTMKernel<float>>(d, act_gate, act_cand, act_cell);
    kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});
    return p;
  }
  return std::dynamic_pointer_cast<LSTMKernel<float>>(kers_.at(key));
}

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
