// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <string>
#include <unordered_set>

#include "paddle/utils/test_macros.h"

namespace paddle {
namespace prim {
class TEST_API PrimCommonUtils {
 public:
  static bool IsBwdPrimEnabled();
  static void SetBwdPrimEnabled(bool enabled);
  static bool IsEagerPrimEnabled();
  static void SetEagerPrimEnabled(bool enabled);
  static bool IsFwdPrimEnabled();
  static void SetFwdPrimEnabled(bool enabled);
  static void SetAllPrimEnabled(bool enabled);
  static size_t CheckSkipCompOps(const std::string& op_type);
  static void AddSkipCompOps(const std::string& op_type);
  static void SetPrimBackwardBlacklist(
      const std::unordered_set<std::string>& op_types);
  static void RemoveSkipCompOps(const std::string& op_type);
  static void SetTargetGradName(const std::map<std::string, std::string>& m);
};
}  // namespace prim
}  // namespace paddle
