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

namespace paddle {
namespace framework {
namespace ir {

class FloatToHalfPass : public FusePassBase {
 public:
  using VarType = framework::proto::VarType;

 public:
  FloatToHalfPass() = default;
  ~FloatToHalfPass() = default;

 protected:
  void ApplyImpl(Graph* graph) const override;

 private:
  void Init(Graph* graph) const;

  void SetDefaultBlacklist() const;

  bool OpSupportPrecision(const std::string& op_type,
                          phi::DataType precision,
                          phi::Backend backend = phi::Backend::GPU) const;

  void SetOpUniqueType() const;

  void RestoreOpOriginType() const;

  inline std::string GetOpOriginalType(const std::string& op_type) const;

  void GetOpPrecision() const;

  void UpdateOpPrecision() const;

  void InsertCastOp() const;

  void ProcessOpWithDtypeAttr() const;

  bool InputVarsNotConvert(Node* op_node, const std::string& var_name) const;

  bool OutputVarsNotConvert(Node* op_node, const std::string& var_name) const;

  void SetVarPrecision() const;

  void ConvertWeightsData() const;

 private:
  mutable bool keep_io_types_;
  // float16 or bfloat16 now
  mutable phi::DataType half_precision_;

  mutable std::unordered_set<std::string> black_list_;

  // subgraph id -> pointer to subgraph
  mutable std::vector<Graph*> subgraphes_;
  // var name -> real var node
  mutable std::unordered_map<std::string, Node*> real_vars_;
  // subgraph id -> all op nodes in subgraph
  mutable std::vector<std::vector<Node*>> all_op_nodes_;
  // op's unique type -> the op's origin type
  mutable std::unordered_map<std::string, std::string> op_original_type_;
  // op's unique type -> whether the op run at half precision
  mutable std::unordered_set<std::string> op_run_half_;

  mutable std::unordered_set<std::string> vars_convert_to_half_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
