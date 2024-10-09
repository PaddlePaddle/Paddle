// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {
struct CastBeforePattern : public PatternBase {
  CastBeforePattern(PDPattern* pattern,
                    const std::string& name_scope,
                    const std::string& mixed_precision_op_type);

  PATTERN_DECL_NODE(cast_in);
  PATTERN_DECL_NODE(cast);
  PATTERN_DECL_NODE(cast_out);
  PATTERN_DECL_NODE(mixed_precision_op);
};

CastBeforePattern::CastBeforePattern(PDPattern* pattern,
                                     const std::string& name_scope,
                                     const std::string& mixed_precision_op_type)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* cast_in =
      pattern->NewNode(cast_in_repr())->assert_is_op_input("cast", "X");
  auto* cast = pattern->NewNode(cast_repr())
                   ->assert_is_op("cast")
                   ->assert_more([&](Node* node) {
                     auto* op_desc = node->Op();
                     return op_desc->GetAttrIfExists<int>("in_dtype") == 5 &&
                            op_desc->GetAttrIfExists<int>("out_dtype") == 4;
                   });
  auto* cast_out = pattern->NewNode(cast_out_repr())
                       ->assert_is_op_output("cast", "Out")
                       ->assert_is_op_input(mixed_precision_op_type, "x")
                       ->assert_has_n_outputs(1);
  auto* mixed_precision_op = pattern->NewNode(mixed_precision_op_repr())
                                 ->assert_is_op(mixed_precision_op_type);

  cast->LinksFrom({cast_in}).LinksTo({cast_out});
  mixed_precision_op->LinksFrom({cast_out});
}

struct CastAfterPattern : public PatternBase {
  CastAfterPattern(PDPattern* pattern,
                   const std::string& name_scope,
                   const std::string& mixed_precision_op_type);

  PATTERN_DECL_NODE(mixed_precision_op);
  PATTERN_DECL_NODE(cast_in);
  PATTERN_DECL_NODE(cast);
  PATTERN_DECL_NODE(cast_out);
};

CastAfterPattern::CastAfterPattern(PDPattern* pattern,
                                   const std::string& name_scope,
                                   const std::string& mixed_precision_op_type)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* mixed_precision_op = pattern->NewNode(mixed_precision_op_repr())
                                 ->assert_is_op(mixed_precision_op_type);
  auto* cast_in = pattern->NewNode(cast_in_repr())
                      ->assert_is_op_output(mixed_precision_op_type, "out")
                      ->assert_is_op_input("cast", "X")
                      ->assert_has_n_outputs(1);
  auto* cast = pattern->NewNode(cast_repr())
                   ->assert_is_op("cast")
                   ->assert_more([&](Node* node) {
                     auto* op_desc = node->Op();
                     return op_desc->GetAttrIfExists<int>("in_dtype") == 4 &&
                            op_desc->GetAttrIfExists<int>("out_dtype") == 5;
                   });
  auto* cast_out =
      pattern->NewNode(cast_out_repr())->assert_is_op_output("cast", "Out");

  mixed_precision_op->LinksTo({cast_in});
  cast->LinksFrom({cast_in}).LinksTo({cast_out});
}

}  // namespace patterns

class CastMixedPrecisionOpFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyCastBeforePass(ir::Graph* graph,
                          const std::string& mixed_precision_op_type) const;
  int ApplyCastAfterPass(ir::Graph* graph,
                         const std::string& mixed_precision_op_type) const;

  const std::string name_scope_{"cast_mixed_precision_op_fuse_pass"};
};

int CastMixedPrecisionOpFusePass::ApplyCastBeforePass(
    ir::Graph* graph, const std::string& mixed_precision_op_type) const {
  GraphPatternDetector gpd;
  patterns::CastBeforePattern pattern(
      gpd.mutable_pattern(), name_scope_, mixed_precision_op_type);
  auto* scope = param_scope();
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastBeforePass";
    GET_IR_NODE(cast_in);
    GET_IR_NODE(cast);
    GET_IR_NODE(cast_out);
    GET_IR_NODE(mixed_precision_op);
    // Note: conv2d_xpu/fc_xpu not support float32/int8/float16, can not fuse.
    if (mixed_precision_op_type == "conv2d_xpu") {
      auto filter_name = mixed_precision_op->Op()->Input("filter")[0];
      auto filter_data_type =
          scope->FindVar(filter_name)->GetMutable<phi::DenseTensor>()->dtype();
      if (filter_data_type == phi::DataType::INT8) {
        return;
      }
    } else if (mixed_precision_op_type == "fc_xpu") {
      auto w_name = mixed_precision_op->Op()->Input("w")[0];
      auto w_data_type =
          scope->FindVar(w_name)->GetMutable<phi::DenseTensor>()->dtype();
      if (w_data_type == phi::DataType::INT8) {
        return;
      }
    }
    mixed_precision_op->Op()->RenameInput(cast_out->Name(), cast_in->Name());
    IR_NODE_LINK_TO(cast_in, mixed_precision_op);

    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {cast, cast_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

int CastMixedPrecisionOpFusePass::ApplyCastAfterPass(
    ir::Graph* graph, const std::string& mixed_precision_op_type) const {
  GraphPatternDetector gpd;
  patterns::CastAfterPattern pattern(
      gpd.mutable_pattern(), name_scope_, mixed_precision_op_type);
  auto* scope = param_scope();
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastAfterPass";
    GET_IR_NODE(mixed_precision_op);
    GET_IR_NODE(cast_in);
    GET_IR_NODE(cast);
    GET_IR_NODE(cast_out);
    // Note: conv2d_xpu/fc_xpu not support float16/int8/float32, can not fuse.
    if (mixed_precision_op_type == "conv2d_xpu") {
      auto filter_name = mixed_precision_op->Op()->Input("filter")[0];
      auto filter_data_type =
          scope->FindVar(filter_name)->GetMutable<phi::DenseTensor>()->dtype();
      auto x_name = mixed_precision_op->Op()->Input("x")[0];
      auto* x_node = FindNodeWithName(graph, x_name);
      if (filter_data_type == phi::DataType::INT8 &&
          x_node->Var()->GetDataType() ==
              proto::VarType::Type::VarType_Type_FP16) {
        return;
      }
    } else if (mixed_precision_op_type == "fc_xpu") {
      auto w_name = mixed_precision_op->Op()->Input("w")[0];
      auto w_data_type =
          scope->FindVar(w_name)->GetMutable<phi::DenseTensor>()->dtype();
      auto x_name = mixed_precision_op->Op()->Input("x")[0];
      auto* x_node = FindNodeWithName(graph, x_name);
      if (w_data_type == phi::DataType::INT8 &&
          x_node->Var()->GetDataType() ==
              proto::VarType::Type::VarType_Type_FP16) {
        return;
      }
    }
    mixed_precision_op->Op()->RenameOutput(cast_in->Name(), cast_out->Name());
    int out_dtype = proto::VarType::Type::VarType_Type_FP32;
    mixed_precision_op->Op()->SetAttr("out_dtype", out_dtype);
    IR_NODE_LINK_TO(mixed_precision_op, cast_out);

    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {cast_in, cast};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void CastMixedPrecisionOpFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int count = 0;
  for (auto op_type : {"conv2d_xpu", "fc_xpu"}) {
    count += ApplyCastBeforePass(graph, op_type);
    count += ApplyCastAfterPass(graph, op_type);
  }
  AddStatis(count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cast_mixed_precision_op_fuse_pass,
              paddle::framework::ir::CastMixedPrecisionOpFusePass);

REGISTER_PASS_CAPABILITY(cast_mixed_precision_op_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "cast", 0));
