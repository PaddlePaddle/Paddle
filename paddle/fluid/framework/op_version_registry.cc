/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace compatible {

OpVersionDesc&& OpVersionDesc::ModifyAttr(const std::string& name,
                                          const std::string& remark,
                                          const OpAttrVariantT& default_value) {
  infos_.push_back(
      std::unique_ptr<OpUpdate<OpAttrInfo, OpUpdateType::kModifyAttr>>(
          new OpUpdate<OpAttrInfo, OpUpdateType::kModifyAttr>(
              OpAttrInfo(name, remark, default_value))));
  return std::move(*this);
}

OpVersionDesc&& OpVersionDesc::NewAttr(const std::string& name,
                                       const std::string& remark,
                                       const OpAttrVariantT& default_value) {
  infos_.push_back(
      std::unique_ptr<OpUpdate<OpAttrInfo, OpUpdateType::kNewAttr>>(
          new OpUpdate<OpAttrInfo, OpUpdateType::kNewAttr>(
              OpAttrInfo(name, remark, default_value))));
  return std::move(*this);
}

OpVersionDesc&& OpVersionDesc::NewInput(const std::string& name,
                                        const std::string& remark) {
  infos_.push_back(
      std::unique_ptr<OpUpdate<OpInputOutputInfo, OpUpdateType::kNewInput>>(
          new OpUpdate<OpInputOutputInfo, OpUpdateType::kNewInput>(
              OpInputOutputInfo(name, remark))));
  return std::move(*this);
}

OpVersionDesc&& OpVersionDesc::NewOutput(const std::string& name,
                                         const std::string& remark) {
  infos_.push_back(
      std::unique_ptr<OpUpdate<OpInputOutputInfo, OpUpdateType::kNewOutput>>(
          new OpUpdate<OpInputOutputInfo, OpUpdateType::kNewOutput>(
              OpInputOutputInfo(name, remark))));
  return std::move(*this);
}

OpVersionDesc&& OpVersionDesc::BugfixWithBehaviorChanged(
    const std::string& remark) {
  infos_.push_back(
      std::unique_ptr<
          OpUpdate<OpBugfixInfo, OpUpdateType::kBugfixWithBehaviorChanged>>(
          new OpUpdate<OpBugfixInfo, OpUpdateType::kBugfixWithBehaviorChanged>(
              OpBugfixInfo(remark))));
  return std::move(*this);
}

OpVersion& OpVersionRegistrar::Register(const std::string& op_type) {
  PADDLE_ENFORCE_EQ(
      op_version_map_.find(op_type), op_version_map_.end(),
      platform::errors::AlreadyExists(
          "'%s' is registered in operator version more than once.", op_type));
  op_version_map_.insert(
      std::pair<std::string, OpVersion>{op_type, OpVersion()});
  return op_version_map_[op_type];
}
uint32_t OpVersionRegistrar::version_id(const std::string& op_type) const {
  auto it = op_version_map_.find(op_type);
  if (it == op_version_map_.end()) {
    return 0;
  }
  return it->second.version_id();
}
}  // namespace compatible
}  // namespace framework
}  // namespace paddle
