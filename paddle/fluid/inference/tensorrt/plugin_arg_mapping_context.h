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

#include <string>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/phi/core/compat/arg_map_context.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class PluginArgumentMappingContext : public ::phi::ArgumentMappingContext {
 public:
  explicit PluginArgumentMappingContext(framework::OpDesc* op_desc_ptr)
      : op_desc_ptr_(op_desc_ptr) {}

  bool HasInput(const std::string& name) const override;

  bool HasOutput(const std::string& name) const override;

  bool HasAttr(const std::string& name) const override;

  paddle::any Attr(const std::string& attr_name) const override;

  size_t InputSize(const std::string& name) const override;

  size_t OutputSize(const std::string& name) const override;

  bool IsDenseTensorInput(const std::string& name) const override;

  bool IsDenseTensorInputs(const std::string& name) const override;

  bool IsSelectedRowsInput(const std::string& name) const override;

  bool IsSelectedRowsInputs(const std::string& name) const override;

  bool IsSparseCooTensorInput(const std::string& name) const override;

  bool IsDenseTensorVectorInput(const std::string& name) const override;

  bool IsDenseTensorOutput(const std::string& name) const override;

  bool IsSelectedRowsOutput(const std::string& name) const override;

  bool IsForInferShape() const override { return false; }

 private:
  framework::OpDesc* op_desc_ptr_;
};
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
