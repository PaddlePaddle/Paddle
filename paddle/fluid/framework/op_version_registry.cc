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

namespace paddle::framework::compatible {

OpVersionDesc&& OpVersionDesc::NewInput(const std::string& name,
                                        const std::string& remark) {
  infos_.emplace_back(
      new_update<OpUpdateType::kNewInput>(OpInputOutputInfo(name, remark)));
  return std::move(*this);
}

OpVersionDesc&& OpVersionDesc::NewOutput(const std::string& name,
                                         const std::string& remark) {
  infos_.emplace_back(
      new_update<OpUpdateType::kNewOutput>(OpInputOutputInfo(name, remark)));
  return std::move(*this);
}

OpVersionDesc&& OpVersionDesc::BugfixWithBehaviorChanged(
    const std::string& remark) {
  infos_.emplace_back(new_update<OpUpdateType::kBugfixWithBehaviorChanged>(
      OpBugfixInfo(remark)));
  return std::move(*this);
}

OpVersionDesc&& OpVersionDesc::DeleteAttr(const std::string& name,
                                          const std::string& remark) {
  infos_.emplace_back(
      new_update<OpUpdateType::kDeleteAttr>(OpAttrInfo(name, remark)));
  return std::move(*this);
}
OpVersionDesc&& OpVersionDesc::ModifyInput(const std::string& name,
                                           const std::string& remark) {
  infos_.emplace_back(
      new_update<OpUpdateType::kModifyInput>(OpInputOutputInfo(name, remark)));
  return std::move(*this);
}
OpVersionDesc&& OpVersionDesc::ModifyOutput(const std::string& name,
                                            const std::string& remark) {
  infos_.emplace_back(
      new_update<OpUpdateType::kModifyOutput>(OpInputOutputInfo(name, remark)));
  return std::move(*this);
}
OpVersionDesc&& OpVersionDesc::DeleteInput(const std::string& name,
                                           const std::string& remark) {
  infos_.emplace_back(
      new_update<OpUpdateType::kDeleteInput>(OpInputOutputInfo(name, remark)));
  return std::move(*this);
}
OpVersionDesc&& OpVersionDesc::DeleteOutput(const std::string& name,
                                            const std::string& remark) {
  infos_.emplace_back(
      new_update<OpUpdateType::kDeleteOutput>(OpInputOutputInfo(name, remark)));
  return std::move(*this);
}

OpVersionRegistrar& OpVersionRegistrar::GetInstance() {
  static OpVersionRegistrar instance;
  return instance;
}

OpVersion& OpVersionRegistrar::Register(const std::string& op_type) {
  PADDLE_ENFORCE_EQ(
      op_version_map_.find(op_type),
      op_version_map_.end(),
      common::errors::AlreadyExists(
          "'%s' is registered in operator version more than once.", op_type));
  op_version_map_.insert(
      std::pair<std::string, OpVersion>{op_type, OpVersion()});
  return op_version_map_[op_type];
}
uint32_t OpVersionRegistrar::version_id(const std::string& op_type) const {
  PADDLE_ENFORCE_NE(
      op_version_map_.count(op_type),
      0,
      common::errors::InvalidArgument(
          "The version of operator type %s has not been registered.", op_type));
  return op_version_map_.find(op_type)->second.version_id();
}

PassVersionCheckerRegistrar& PassVersionCheckerRegistrar::GetInstance() {
  static PassVersionCheckerRegistrar instance;
  return instance;
}

// Provide a fake registration item for pybind testing.
#include "paddle/fluid/framework/op_version_registry.inl"

}  // namespace paddle::framework::compatible
