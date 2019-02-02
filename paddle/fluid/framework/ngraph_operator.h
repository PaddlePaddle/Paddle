/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/variant.h"

#include "ngraph/type/element_type.hpp"

namespace paddle {
namespace framework {

class NgraphOperator : public OperatorBase {
 public:
  static std::vector<
      std::vector<std::vector<std::unique_ptr<OperatorBase>>::iterator>>
  NgraphOpIntervals(
      std::vector<std::unique_ptr<paddle::framework::OperatorBase>>* ops);

  explicit NgraphOperator(
      const ProgramDesc& prog, size_t block_id,
      std::vector<std::unique_ptr<OperatorBase>>::iterator start,
      std::vector<std::unique_ptr<OperatorBase>>::iterator end,
      const std::string& type = "fused_op", const VariableNameMap& inputs = {},
      const VariableNameMap& outputs = {}, const AttributeMap& attrs = {});

  void RunImpl(const Scope& scope, const platform::Place& place) const final;

 private:
  const ProgramDesc pdesc_;
  size_t block_;
  std::vector<std::shared_ptr<OperatorBase>> fused_ops_;
  std::unordered_map<std::string, ngraph::element::Type> var_type_map_;
  std::unordered_set<std::string> persistables_;
  std::unordered_set<std::string> fetches_;
  std::unordered_set<std::string> post_op_inputs_;
  bool is_full_ = false;

  void Process();
};
}  // namespace framework
}  // namespace paddle
