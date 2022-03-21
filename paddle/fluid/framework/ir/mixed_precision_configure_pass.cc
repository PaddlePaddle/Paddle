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

#include "paddle/fluid/framework/ir/mixed_precision_configure_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

void MixedPrecisionConfigurePass::InsertCastOps(
    Graph* graph, const StringSet& blacklist) const {
  VLOG(3) << "Insert the cast op before and after the kernel that does not "
             "supports fp16 precision";

  auto update_cast_desc = [&](
      framework::OpDesc& desc, const std::string& x_name,
      const std::string& out_name, const int in_dtype, const int out_dtype) {
    desc.SetType("cast");
    desc.SetInput("X", {x_name});
    desc.SetOutput("Out", {out_name});
    desc.SetAttr("in_dtype", in_dtype);
    desc.SetAttr("out_dtype", out_dtype);
    desc.SetAttr("use_mkldnn", false);
    desc.SetAttr("with_quant_attr", false);
    desc.Flush();
  };

  auto cast_input = [&](Graph* graph, Node* op_node,
                        const StringSet& cast_list) {
    auto inlinks = op_node->inputs;
    for (auto* pre_node : inlinks) {
      if (pre_node->IsVar()) {
        const auto is_persistable = pre_node->Var()->Persistable();
        const auto is_float =
            pre_node->Var()->GetDataType() == proto::VarType::FP16 ||
            pre_node->Var()->GetDataType() == proto::VarType::FP32 ||
            pre_node->Var()->GetDataType() == proto::VarType::FP64;
        if (!is_persistable && is_float) {
          int suffix = 0;
          for (auto* pre_node_input : pre_node->inputs) {
            if (!pre_node_input->IsOp()) continue;
            const auto& type = pre_node_input->Op()->Type();
            if (!cast_list.count(type) && type != "cast") {
              std::string old_name = pre_node->Name();
              std::string new_name =
                  old_name + "_cast.tmp_" + std::to_string(suffix);
              suffix++;

              framework::OpDesc new_op_desc(op_node->Op()->Block());
              // 4 for fp16, 5 for fp32
              update_cast_desc(new_op_desc, old_name, new_name, 4, 5);
              auto* new_op = graph->CreateOpNode(&new_op_desc);

              VarDesc out_var(new_name);
              out_var.SetPersistable(false);
              auto* node_var = graph->CreateVarNode(&out_var);

              op_node->Op()->RenameInput(old_name, new_name);
              IR_NODE_LINK_TO(pre_node, new_op);
              IR_NODE_LINK_TO(new_op, node_var);
              IR_NODE_LINK_TO(node_var, op_node);
            }
          }
        }
      }
    }
  };

  auto cast_output = [&](Graph* graph, Node* op_node,
                         const StringSet& cast_list) {
    auto outlinks = op_node->outputs;
    for (auto* next_node : outlinks) {
      if (next_node->IsVar()) {
        const auto is_persistable = next_node->Var()->Persistable();
        const auto is_float =
            next_node->Var()->GetDataType() == proto::VarType::FP16 ||
            next_node->Var()->GetDataType() == proto::VarType::FP32 ||
            next_node->Var()->GetDataType() == proto::VarType::FP64;
        if (!is_persistable && is_float) {
          int suffix = 0;
          for (auto* next_node_output : next_node->outputs) {
            if (!next_node_output->IsOp()) continue;

            const auto& type = next_node_output->Op()->Type();
            if (!cast_list.count(type) && type != "cast") {
              std::string old_name = next_node->Name();
              std::string new_name =
                  old_name + "_cast.tmp_" + std::to_string(suffix);
              suffix++;

              framework::OpDesc new_op_desc(op_node->Op()->Block());
              // 4 for fp16, 5 for fp32
              update_cast_desc(new_op_desc, old_name, new_name, 5, 4);
              auto* new_op = graph->CreateOpNode(&new_op_desc);

              VarDesc out_var(new_name);
              out_var.SetPersistable(false);
              auto* node_var = graph->CreateVarNode(&out_var);

              next_node_output->Op()->RenameInput(old_name, new_name);
              IR_NODE_LINK_TO(next_node, new_op);
              IR_NODE_LINK_TO(new_op, node_var);
              IR_NODE_LINK_TO(node_var, next_node_output);
            }
          }
        }
      }
    }
  };

  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp() || op_node->Op()->Type() == "feed" ||
        op_node->Op()->Type() == "fetch")
      continue;

    const auto& type = op_node->Op()->Type();
    if (blacklist.count(type)) {
      cast_input(graph, op_node, blacklist);
      cast_output(graph, op_node, blacklist);
    }
  }
}

void MixedPrecisionConfigurePass::ApplyImpl(Graph* graph) const {
  const auto blacklist =
      Get<std::unordered_set<std::string>>("gpu_fp16_disabled_op_types");
  InsertCastOps(graph, blacklist);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mixed_precision_configure_pass,
              paddle::framework::ir::MixedPrecisionConfigurePass);
