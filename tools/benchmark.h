//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
/**
 * @brief benchmark the Kernel performance on single thread single GPU.
 */

// Get current time point in ns.
inline uint64_t NanoTime();

template <typename DeviceContext>
class Benchmark {
 public:
  explicit Benchmark(const char* name) : name_(name) {}
  void Register(const char* op);
  void Run(int iters) const;
  // void RunRepeats() const;

 private:
  std::string name_;
  std::vector<std::unique_ptr<OperatorBase>> ops_;
};

// NOTE(dzhwinter): benchmark only support OpWithkernel
#define TEST_OP_CPU(op_name, iters)                    \
  USE_OP(#op_name);                                    \
  PADDLE_ENFORCE(OpInfoMap::Instance().Has(#op_name)); \
  Benchmark<platform::CPUDeviceContext> bench(#op_name);

}  // namespace framework
}  // namespace paddle
