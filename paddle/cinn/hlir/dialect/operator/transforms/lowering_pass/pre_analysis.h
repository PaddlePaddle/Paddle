// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>
#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"

namespace cinn::dialect::ir::details {
using OpLoweringGroup = cinn::hlir::framework::pir::OpLoweringGroup;
using OpLoweringGroupPtr = std::shared_ptr<OpLoweringGroup>;
using GroupInfoMap = std::unordered_map<::pir::Operation*, OpLoweringGroupPtr>;

class FusionOpAnalysis final {
 public:
  FusionOpAnalysis(GroupInfoMap* group_infos, bool is_dy_shape)
      : group_infos_(group_infos), is_dy_shape_(is_dy_shape) {}
  void Run(pir::Operation* module_op) {
    RunImpl(module_op);
    PreCompileGroup();
  }

 protected:
  void RunImpl(pir::Operation* op);
  void GatherGroup(pir::Operation* fusion_op);
  void PreCompileGroup();

 private:
  GroupInfoMap* group_infos_;  // not_owned
  bool is_dy_shape_;
};
}  // namespace cinn::dialect::ir::details
