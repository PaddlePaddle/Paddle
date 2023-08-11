// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/scope.h"
#include "paddle/cinn/hlir/framework/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {

enum CheckResult { kOK = 0, kZero = 1, kNaN = 2, kInf = 3, kOne = 4 };

class AccuracyChecker {
 public:
  AccuracyChecker(const Target& target, Scope* scope)
      : target_(target), scope_(scope) {}

  std::string operator()(const std::string& arg_name);
  std::string operator()(
      const std::map<std::string, cinn_pod_value_t>* name2podargs,
      const std::string& arg_name);

 private:
  template <typename T>
  std::string CheckTensor(const Tensor& tensor, const std::string& arg_name);

  template <typename T>
  std::string CheckBuffer(const cinn_buffer_t* buffer,
                          const std::string& arg_name);

  template <typename T>
  void MemcpyDeviceToHost(const T* src, size_t numel, T* dst);

  template <typename T>
  CheckResult CheckNanOrInf(const Tensor& cpu_tensor);

  Target target_;
  Scope* scope_;  // Not owned
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
