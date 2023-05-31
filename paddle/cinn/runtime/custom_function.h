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

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace runtime {

namespace utils {
class AssertTrueMsgTool {
 public:
  static AssertTrueMsgTool* GetInstance() {
    static AssertTrueMsgTool msg;
    return &msg;
  }

  void SetMsg(int key, const std::string& msg);
  const std::string& GetMsg(int key);

  bool FindFlag(const std::string& param) { return flag_values_.count(param); }

  template <typename T>
  const T& GetFlagValue(const std::string& param) {
    InitFlagInfo();
    CHECK(flag_values_.count(param))
        << "The FLAGS_cinn_check_fusion_accuracy_pass only support parameter "
           "\"only_warning/rtol/atol/equal_nan\" now";
    CHECK(absl::holds_alternative<T>(flag_values_.at(param)))
        << "Try get value from a error type!";
    return absl::get<T>(flag_values_.at(param));
  }

 private:
  AssertTrueMsgTool() = default;

  void InitFlagInfo();

  std::unordered_map<std::string, cinn::utils::Attribute> flag_values_;
  std::unordered_map<int, std::string> global_msg_;

  CINN_DISALLOW_COPY_AND_ASSIGN(AssertTrueMsgTool);
};
}  // namespace utils

void cinn_assert_true(void* v_args,
                      int num_args,
                      int msg,
                      bool only_warning,
                      void* stream,
                      const Target& target);

}  // namespace runtime
}  // namespace cinn
