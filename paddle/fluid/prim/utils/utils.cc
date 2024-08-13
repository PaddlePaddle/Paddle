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
#include "paddle/common/flags.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"

PHI_DEFINE_EXPORTED_bool(prim_enabled, false, "enable_prim or not");
PHI_DEFINE_EXPORTED_bool(prim_all, false, "enable prim_all or not");
PHI_DEFINE_EXPORTED_bool(prim_forward, false, "enable prim_forward or not");
PHI_DEFINE_EXPORTED_bool(prim_backward, false, "enable prim_backward not");

namespace paddle::prim {
bool PrimCommonUtils::IsBwdPrimEnabled() {
  bool res = StaticCompositeContext::Instance().IsBwdPrimEnabled();
  return res || FLAGS_prim_all || FLAGS_prim_backward;
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
  bool res = StaticCompositeContext::Instance().IsFwdPrimEnabled();
  return res || FLAGS_prim_all || FLAGS_prim_forward;
}

void PrimCommonUtils::SetFwdPrimEnabled(bool enable_prim) {
  StaticCompositeContext::Instance().SetFwdPrimEnabled(enable_prim);
}

void PrimCommonUtils::SetAllPrimEnabled(bool enable_prim) {
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

}  // namespace paddle::prim
