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

#include <gtest/gtest.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {
namespace tests {

class TestArgumentMappingContext : public phi::ArgumentMappingContext {
 public:
  TestArgumentMappingContext(
      std::unordered_set<std::string> dense_tensor_ins,
      std::unordered_set<std::string> sr_ins,
      std::unordered_map<std::string, paddle::any> op_attrs,
      std::unordered_set<std::string> dense_tensor_outs,
      std::unordered_set<std::string> sr_outs = {})
      : dense_tensor_inputs(dense_tensor_ins),
        selected_rows_inputs(sr_ins),
        attrs(op_attrs),
        dense_tensor_outputs(dense_tensor_outs),
        selected_rows_outputs(sr_outs) {}

  bool HasInput(const std::string& name) const override {
    return dense_tensor_inputs.count(name) > 0 ||
           selected_rows_inputs.count(name) > 0;
  }

  bool HasOutput(const std::string& name) const override {
    return dense_tensor_outputs.count(name) > 0 ||
           selected_rows_outputs.count(name) > 0;
  }

  bool HasAttr(const std::string& name) const override {
    return attrs.count(name) > 0;
  }

  paddle::any Attr(const std::string& name) const override {
    return attrs.at(name);
  }

  size_t InputSize(const std::string& name) const override {
    return dense_tensor_inputs.size() + selected_rows_inputs.size();
  }

  size_t OutputSize(const std::string& name) const override {
    return dense_tensor_outputs.size() + selected_rows_outputs.size();
  }

  bool IsDenseTensorInput(const std::string& name) const override {
    return dense_tensor_inputs.count(name) > 0;
  }

  bool IsSelectedRowsInput(const std::string& name) const override {
    return selected_rows_inputs.count(name) > 0;
  }

  // add member if needed
  bool IsDenseTensorVectorInput(const std::string& name) const override {
    return false;
  }

  bool IsDenseTensorOutput(const std::string& name) const override {
    return dense_tensor_outputs.count(name) > 0;
  }

  bool IsSelectedRowsOutput(const std::string& name) const override {
    return selected_rows_outputs.count(name) > 0;
  }

  bool IsForInferShape() const override { return false; }

 private:
  const std::unordered_set<std::string> dense_tensor_inputs;
  const std::unordered_set<std::string> selected_rows_inputs;
  const std::unordered_map<std::string, paddle::any> attrs;
  const std::unordered_set<std::string> dense_tensor_outputs;
  const std::unordered_set<std::string> selected_rows_outputs;
};

}  // namespace tests
}  // namespace phi
