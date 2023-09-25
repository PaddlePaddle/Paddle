// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/cinn/common/context.h"
#include "paddle/pir/core/operation.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace newir {

struct CompatibleInfo {
  static constexpr char* kNamePrefix = "var_";
  // TODO(Aurelius): Need add name mapping logic in REGISTER_CINN_OP
  // macros or attempt to unify Op name with Paddle and CINN.
  static const std::unordered_map<std::string, std::string> OP_NAMES;

  static std::string OpName(const ::pir::Operation& op);

  static std::string ValueName(const ::pir::Value& value);

  static std::string OpFuncName(const ::pir::Operation& op);

  static std::string GroupOpsName(const std::vector<::pir::Operation*>& ops);

  static std::vector<std::string> InputNames(const ::pir::Operation& op,
                                             bool allow_duplicate = false);

  static std::vector<std::string> OutputNames(::pir::Operation& op);  // NOLINT
};

}  // namespace newir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
