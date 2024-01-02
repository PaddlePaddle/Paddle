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

#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/fluid/platform/flags.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"

PADDLE_DEFINE_EXPORTED_bool(prim_enabled, false, "enable_prim or not");
PADDLE_DEFINE_EXPORTED_string(prim_blacklist, "", "prim ops blacklist");

namespace paddle {
namespace prim {

bool PrimCommonUtils::IsBwdPrimEnabled() {
  return StaticCompositeContext::Instance().IsBwdPrimEnabled();
}

void PrimCommonUtils::SetBwdPrimEnabled(bool enable_prim) {
  StaticCompositeContext::Instance().SetBwdPrimEnabled(enable_prim);
}

bool PrimCommonUtils::IsEagerPrimEnabled() {
  return StaticCompositeContext::Instance().IsEagerPrimEnabled();
}

void PrimCommonUtils::SetEagerPrimEnabled(bool enable_prim) {
  StaticCompositeContext::Instance().SetEagerPrimEnabled(enable_prim);
}

bool PrimCommonUtils::IsFwdPrimEnabled() {
  return StaticCompositeContext::Instance().IsFwdPrimEnabled();
}

void PrimCommonUtils::SetFwdPrimEnabled(bool enable_prim) {
  VLOG(0) << "FLAGS_prim_enabled ====================== " << FLAGS_prim_enabled;
  StaticCompositeContext::Instance().SetFwdPrimEnabled(enable_prim);
}

void PrimCommonUtils::SetAllPrimEnabled(bool enable_prim) {
  VLOG(0) << "FLAGS_prim_enabled ====================== " << FLAGS_prim_enabled;
  StaticCompositeContext::Instance().SetAllPrimEnabled(enable_prim);
}

size_t PrimCommonUtils::CheckSkipCompOps(const std::string& op_type) {
  return StaticCompositeContext::Instance().CheckSkipCompOps(op_type);
}

void PrimCommonUtils::AddSkipCompOps(const std::string& op_type) {
  StaticCompositeContext::Instance().AddSkipCompOps(op_type);
}

void PrimCommonUtils::SetPrimBackwardBlacklist(
    const std::unordered_set<std::string>& op_types) {
  for (const auto& item : op_types) {
    StaticCompositeContext::Instance().AddSkipCompOps(item);
  }
}

void PrimCommonUtils::RemoveSkipCompOps(const std::string& op_type) {
  StaticCompositeContext::Instance().RemoveSkipCompOps(op_type);
}

void PrimCommonUtils::SetTargetGradName(
    const std::map<std::string, std::string>& m) {
  StaticCompositeContext::Instance().SetTargetGradName(m);
}

}  // namespace prim
}  // namespace paddle
