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

class AutoMixedPrecisionPass : public FusePassBase {
 public:
  using VarType = framework::proto::VarType;

 public:
  AutoMixedPrecisionPass() = default;
  ~AutoMixedPrecisionPass() = default;

 protected:
  void ApplyImpl(Graph* graph) const override;

 private:
  void Init(Graph* graph) const;

  void SetDefaultBlacklist() const;

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
  mutable bool skip_pass_{false};

  mutable bool keep_io_types_{false};
  // float16 or bfloat16 now
  mutable phi::DataType low_precision_{phi::DataType::FLOAT16};

  mutable phi::Backend backend_{phi::Backend::GPU};

  mutable std::unordered_set<std::string> black_list_;

  // subgraph id -> pointer to subgraph
  mutable std::vector<Graph*> subgraphes_;
  // var name -> real var node
  mutable std::unordered_map<std::string, Node*> real_vars_;
  // subgraph id -> all op nodes in subgraph
  mutable std::vector<std::vector<Node*>> all_op_nodes_;
  // op's unique type -> the op's origin type
  mutable std::unordered_map<std::string, std::string> op_original_type_;
  // op's unique type -> whether the op run at low precision
  mutable std::unordered_set<std::string> op_run_low_precision_;

  mutable std::unordered_set<std::string> vars_convert_to_low_precision_;
};

bool OpSupportPrecision(const std::string& op_type,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& black_list);

void DoInsertCastOp(Graph* graph,
                    Node* var_node,
                    Node* op_node,
                    proto::VarType::Type from_type,
                    proto::VarType::Type to_type,
                    framework::BlockDesc* block_desc,
                    int* suffix,
                    std::unordered_map<Node*, Node*>* cache);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
