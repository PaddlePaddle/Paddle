/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <mlir/IR/Operation.h>
#include <unordered_map>
#include "paddle/infrt/dialect/pd/common/pd_ops_info.h"
#include "paddle/phi/core/compat/arg_map_context.h"

namespace infrt {
class ProtoArgumentMappingContext : public ::phi::ArgumentMappingContext {
 public:
  // only support op in pd dialect
  explicit ProtoArgumentMappingContext(mlir::Operation* op)
      : op_(op),
        input_map_(pd_dialect_inputs_info_map_.at(
            op->getName().getIdentifier().str().substr(3))),
        output_map_(pd_dialect_outputs_info_map_.at(
            op->getName().getIdentifier().str().substr(3))) {}
  bool HasInput(const std::string& name) const override;
  bool HasOutput(const std::string& name) const override;
  bool HasAttr(const std::string& name) const override;

  // now we can't use Attribute here, it will cause phi relay on
  // boost::variant and BlockDesc
  paddle::any Attr(const std::string& name) const override;

  size_t InputSize(const std::string& name) const override;
  size_t OutputSize(const std::string& name) const override;

  bool IsDenseTensorInput(const std::string& name) const override;
  bool IsSelectedRowsInput(const std::string& name) const override;
  bool IsDenseTensorVectorInput(const std::string& name) const override;

  bool IsDenseTensorOutput(const std::string& name) const override;
  bool IsSelectedRowsOutput(const std::string& name) const override;

  bool IsForInferShape() const override { return false; }

 private:
  mlir::Operation* op_;
  const std::unordered_map<std::string, uint8_t>& input_map_;
  const std::unordered_map<std::string, uint8_t>& output_map_;
};

}  // namespace infrt
