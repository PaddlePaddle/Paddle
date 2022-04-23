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

#include "paddle/infrt/dialect/phi/pass/proto_arg_map_context.h"

namespace infrt {

bool ProtoArgumentMappingContext::HasInput(const std::string& name) const {
  if (input_map_.find(name) == input_map_.end()) {
    return false;
  }
  uint8_t index = input_map_.at(name);
  return static_cast<bool>(op_->getOperand(index));
}

bool ProtoArgumentMappingContext::HasOutput(const std::string& name) const {
  if (output_map_.find(name) == output_map_.end()) {
    return false;
  }
  return true;
}

bool ProtoArgumentMappingContext::HasAttr(const std::string& name) const {
  if (name == "is_test") return true;
  return op_->hasAttr(name);
}

paddle::any ProtoArgumentMappingContext::Attr(const std::string& name) const {
  if (name == "is_test") {
    return paddle::any(true);
  }
  mlir::Attribute attr = op_->getAttr(name);
  if (!attr) {
    return paddle::any();
  }
  if (mlir::StringAttr str_attr = attr.dyn_cast<mlir::StringAttr>()) {
    return paddle::any(str_attr.str());
  }

  // ToDO: implementation in the ext PR.
  return paddle::any(0);
}

size_t ProtoArgumentMappingContext::InputSize(const std::string& name) const {
  return op_->getNumOperands();
}
size_t ProtoArgumentMappingContext::OutputSize(const std::string& name) const {
  return op_->getNumResults();
}

bool ProtoArgumentMappingContext::IsDenseTensorInput(
    const std::string& name) const {
  return true;
}
bool ProtoArgumentMappingContext::IsSelectedRowsInput(
    const std::string& name) const {
  return false;
}
bool ProtoArgumentMappingContext::IsDenseTensorVectorInput(
    const std::string& name) const {
  return false;
}

bool ProtoArgumentMappingContext::IsDenseTensorOutput(
    const std::string& name) const {
  return true;
}
bool ProtoArgumentMappingContext::IsSelectedRowsOutput(
    const std::string& name) const {
  return false;
}

}  // namespace infrt
