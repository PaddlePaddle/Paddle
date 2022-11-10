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
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace framework {
namespace ir {

class FloatToMixedPass : public FusePassBase {
 public:
  using VarType = framework::proto::VarType;

 public:
  FloatToMixedPass() = default;
  ~FloatToMixedPass() = default;

 protected:
  void ApplyImpl(framework::ir::Graph* graph) const override;

 private:
  void Init(framework::ir::Graph* graph) const;

  bool OpSupportPrecision(const std::string& op_type,
                          phi::DataType precision,
                          phi::Backend backend = phi::Backend::GPU) const;

  void SetOpUniqueType() const;

  void RestoreOpOriginType() const;

  void GetVarInputOps() const;

  void GetOpPrecision() const;

  void SetVarAndUpdateOpPrecision() const;

  void InsertCastOp() const;

  void ProcessOpWithDtypeAttr() const;

  void ProcessWeights() const;

  bool WeightsNotMixed(framework::ir::Node* var_node) const;

 private:
  mutable bool keep_io_types_;
  // float16 or bfloat16 now
  mutable phi::DataType mixed_precision_;

  mutable std::unordered_set<std::string> blacklist_;

  // subgraph id -> pointer to subgraph
  mutable std::vector<framework::ir::Graph*> subgraphes_;
  // var name -> real var node
  mutable std::unordered_map<std::string, framework::ir::Node*> real_vars_;
  // subgraph id -> all nodes in subgraph
  mutable std::vector<std::unordered_set<framework::ir::Node*>> all_nodes_;
  // op's unique type -> the op's origin type
  mutable std::unordered_map<std::string, std::string> op_original_type_;
  // op's unique type -> whether the op run at mixed precision
  mutable std::unordered_map<std::string, bool> op_run_mixed_;
  // var -> the var's all input op
  mutable std::unordered_map<std::string, std::vector<framework::ir::Node*>>
      var_input_ops_;
  mutable std::unordered_set<std::string> weights_should_be_mixed_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
