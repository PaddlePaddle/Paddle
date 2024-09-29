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

#include "paddle/fluid/framework/ir/xpu/one_beam_size_fuse_pass.h"
#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct AssignPattern : public PatternBase {
  AssignPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(assign);
  // declare variable node's name
  PATTERN_DECL_NODE(assign_out);
};

AssignPattern::AssignPattern(PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* assign =
      pattern->NewNode(assign_repr())
          ->assert_is_op("assign")
          ->assert_more([&](Node* node) {
            auto pre_op_nodes = node->inputs[0]->inputs;
            return pre_op_nodes.size() == 1 &&
                   pre_op_nodes[0]->Op()->Type() == "fused_multi_transformer";
          });
  auto* assign_out =
      pattern->NewNode(assign_out_repr())->assert_is_op_output("assign", "Out");

  assign->LinksTo({assign_out});
}

struct ShapeAssociatedOpsPattern : public PatternBase {
  ShapeAssociatedOpsPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(shape);
  PATTERN_DECL_NODE(slice);
  PATTERN_DECL_NODE(div);
  PATTERN_DECL_NODE(cast_0);
  PATTERN_DECL_NODE(cast_1);
  PATTERN_DECL_NODE(scale_0);
  PATTERN_DECL_NODE(cast_2);
  PATTERN_DECL_NODE(range);
  PATTERN_DECL_NODE(unsqueeze2);
  PATTERN_DECL_NODE(scale_1);
  PATTERN_DECL_NODE(add);
  PATTERN_DECL_NODE(flatten_contiguous_range);
  // declare variable node's name
  PATTERN_DECL_NODE(shape_out);
  PATTERN_DECL_NODE(slice_out);
  PATTERN_DECL_NODE(div_out);
  PATTERN_DECL_NODE(cast_0_out);
  PATTERN_DECL_NODE(cast_1_out);
  PATTERN_DECL_NODE(scale_0_out);
  PATTERN_DECL_NODE(cast_2_out);
  PATTERN_DECL_NODE(range_out);
  PATTERN_DECL_NODE(unsqueeze2_out);
  PATTERN_DECL_NODE(scale_1_out);
  PATTERN_DECL_NODE(add_x);
  PATTERN_DECL_NODE(add_out);
};

ShapeAssociatedOpsPattern::ShapeAssociatedOpsPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* shape = pattern->NewNode(shape_repr())->assert_is_op("shape");
  auto* shape_out = pattern->NewNode(shape_out_repr())
                        ->assert_is_op_output("shape", "Out")
                        ->assert_is_op_input("slice", "Input");
  auto* slice =
      pattern->NewNode(slice_repr())
          ->assert_is_op("slice")
          ->assert_more([&](Node* node) {
            auto* op_desc = node->Op();
            return op_desc->GetAttrIfExists<std::vector<int>>("axes") ==
                       std::vector<int>{0} &&
                   op_desc->GetAttrIfExists<std::vector<int>>("starts") ==
                       std::vector<int>{0} &&
                   op_desc->GetAttrIfExists<std::vector<int>>("ends") ==
                       std::vector<int>{1};
          });
  auto* slice_out = pattern->NewNode(slice_out_repr())
                        ->assert_is_op_output("slice", "Out")
                        ->assert_is_op_input("elementwise_div", "X")
                        ->assert_is_op_input("elementwise_div", "Y")
                        ->assert_is_op_input("cast", "X")
                        ->assert_is_op_input("scale", "X");
  auto* div = pattern->NewNode(div_repr())->assert_is_op("elementwise_div");
  auto* div_out = pattern->NewNode(div_out_repr())
                      ->assert_is_op_output("elementwise_div", "Out")
                      ->assert_is_op_input("cast", "X");
  auto* cast_0 = pattern->NewNode(cast_0_repr())->assert_is_op("cast");
  auto* cast_0_out = pattern->NewNode(cast_0_out_repr())
                         ->assert_is_op_output("cast", "Out")
                         ->assert_is_op_input("range", "Step");
  auto* cast_1 = pattern->NewNode(cast_1_repr())->assert_is_op("cast");
  auto* cast_1_out = pattern->NewNode(cast_1_out_repr())
                         ->assert_is_op_output("cast", "Out")
                         ->assert_is_op_input("range", "End");
  auto* scale_0 = pattern->NewNode(scale_0_repr())->assert_is_op("scale");
  auto* scale_0_out = pattern->NewNode(scale_0_out_repr())
                          ->assert_is_op_output("scale", "Out")
                          ->assert_is_op_input("cast", "X");
  auto* cast_2 = pattern->NewNode(cast_2_repr())->assert_is_op("cast");
  auto* cast_2_out = pattern->NewNode(cast_2_out_repr())
                         ->assert_is_op_output("cast", "Out")
                         ->assert_is_op_input("range", "Start");
  auto* range = pattern->NewNode(range_repr())->assert_is_op("range");
  auto* range_out = pattern->NewNode(range_out_repr())
                        ->assert_is_op_output("range", "Out")
                        ->assert_is_op_input("unsqueeze2", "X");
  auto* unsqueeze2 =
      pattern->NewNode(unsqueeze2_repr())->assert_is_op("unsqueeze2");
  auto* unsqueeze2_out = pattern->NewNode(unsqueeze2_out_repr())
                             ->assert_is_op_output("unsqueeze2", "Out")
                             ->assert_is_op_input("scale", "X");
  auto* scale_1 = pattern->NewNode(scale_1_repr())->assert_is_op("scale");
  auto* scale_1_out = pattern->NewNode(scale_1_out_repr())
                          ->assert_is_op_output("scale", "Out")
                          ->assert_is_op_input("elementwise_add", "Y");
  auto* add_x = pattern->NewNode(add_x_repr())
                    ->assert_is_op_input("elementwise_add", "X");
  auto* add = pattern->NewNode(add_repr())->assert_is_op("elementwise_add");
  auto* add_out = pattern->NewNode(add_out_repr())
                      ->assert_is_op_output("elementwise_add", "Out")
                      ->assert_is_op_input("flatten_contiguous_range", "X");
  auto* flatten_contiguous_range =
      pattern->NewNode(flatten_contiguous_range_repr())
          ->assert_is_op("flatten_contiguous_range");

  shape->LinksTo({shape_out});
  slice->LinksFrom({shape_out}).LinksTo({slice_out});
  div->LinksFrom({slice_out}).LinksTo({div_out});
  cast_0->LinksFrom({div_out}).LinksTo({cast_0_out});
  cast_1->LinksFrom({slice_out}).LinksTo({cast_1_out});
  scale_0->LinksFrom({slice_out}).LinksTo({scale_0_out});
  cast_2->LinksFrom({scale_0_out}).LinksTo({cast_2_out});
  range->LinksFrom({cast_0_out, cast_1_out, cast_2_out}).LinksTo({range_out});
  unsqueeze2->LinksFrom({range_out}).LinksTo({unsqueeze2_out});
  scale_1->LinksFrom({unsqueeze2_out}).LinksTo({scale_1_out});
  add->LinksFrom({scale_1_out, add_x}).LinksTo({add_out});
  flatten_contiguous_range->LinksFrom({add_out});
}

struct BeamSearchAssociatedOpsPattern : public PatternBase {
  BeamSearchAssociatedOpsPattern(PDPattern* pattern,
                                 const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(lod_reset_0);
  PATTERN_DECL_NODE(lod_reset_1);
  PATTERN_DECL_NODE(beam_search);
  PATTERN_DECL_NODE(write_to_array_0);
  PATTERN_DECL_NODE(write_to_array_1);
  PATTERN_DECL_NODE(is_empty);
  PATTERN_DECL_NODE(logical_not);
  PATTERN_DECL_NODE(cast);
  // declare variable node's name
  PATTERN_DECL_NODE(lod_reset_0_out);
  PATTERN_DECL_NODE(lod_reset_1_out);
  PATTERN_DECL_NODE(beam_search_parent_idx);
  PATTERN_DECL_NODE(beam_search_selected_ids);
  PATTERN_DECL_NODE(beam_search_selected_scores);
  PATTERN_DECL_NODE(is_empty_out);
  PATTERN_DECL_NODE(logical_not_out);
  PATTERN_DECL_NODE(cast_out);
};

BeamSearchAssociatedOpsPattern::BeamSearchAssociatedOpsPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* lod_reset_0 =
      pattern->NewNode(lod_reset_0_repr())->assert_is_op("lod_reset");
  auto* lod_reset_0_out = pattern->NewNode(lod_reset_0_out_repr())
                              ->assert_is_op_output("lod_reset", "Out")
                              ->assert_is_op_input("beam_search", "ids");
  auto* lod_reset_1 =
      pattern->NewNode(lod_reset_1_repr())->assert_is_op("lod_reset");
  auto* lod_reset_1_out = pattern->NewNode(lod_reset_1_out_repr())
                              ->assert_is_op_output("lod_reset", "Out")
                              ->assert_is_op_input("beam_search", "scores");
  auto* beam_search =
      pattern->NewNode(beam_search_repr())->assert_is_op("beam_search");
  auto* beam_search_selected_ids =
      pattern->NewNode(beam_search_selected_ids_repr())
          ->assert_is_op_output("beam_search", "selected_ids")
          ->assert_is_op_input("write_to_array", "X")
          ->assert_is_op_input("is_empty", "X");
  auto* beam_search_selected_scores =
      pattern->NewNode(beam_search_selected_scores_repr())
          ->assert_is_op_output("beam_search", "selected_scores")
          ->assert_is_op_input("write_to_array", "X");
  auto* beam_search_parent_idx =
      pattern->NewNode(beam_search_parent_idx_repr())
          ->assert_is_op_output("beam_search", "parent_idx")
          ->assert_is_op_input("cast", "X");
  auto* write_to_array_0 =
      pattern->NewNode(write_to_array_0_repr())->assert_is_op("write_to_array");
  auto* write_to_array_1 =
      pattern->NewNode(write_to_array_1_repr())->assert_is_op("write_to_array");
  auto* is_empty = pattern->NewNode(is_empty_repr())->assert_is_op("is_empty");
  auto* is_empty_out = pattern->NewNode(is_empty_out_repr())
                           ->assert_is_op_output("is_empty", "Out")
                           ->assert_is_op_input("logical_not", "X");
  auto* logical_not =
      pattern->NewNode(logical_not_repr())->assert_is_op("logical_not");
  auto* logical_not_out = pattern->NewNode(logical_not_out_repr())
                              ->assert_is_op_output("logical_not", "Out");
  auto* cast = pattern->NewNode(cast_repr())->assert_is_op("cast");
  auto* cast_out =
      pattern->NewNode(cast_out_repr())->assert_is_op_output("cast", "Out");

  lod_reset_0->LinksTo({lod_reset_0_out});
  lod_reset_1->LinksTo({lod_reset_1_out});
  beam_search->LinksFrom({lod_reset_0_out, lod_reset_1_out})
      .LinksTo({beam_search_selected_ids,
                beam_search_selected_scores,
                beam_search_parent_idx});
  write_to_array_0->LinksFrom({beam_search_selected_ids});
  write_to_array_1->LinksFrom({beam_search_selected_scores});
  is_empty->LinksFrom({beam_search_selected_ids}).LinksTo({is_empty_out});
  logical_not->LinksFrom({is_empty_out}).LinksTo({logical_not_out});
  cast->LinksFrom({beam_search_parent_idx}).LinksTo({cast_out});
}

}  // namespace patterns

bool OnlyOneBeamSearchAndOneBeamSize(ir::Graph* graph) {
  std::vector<Node*> beam_search_nodes;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "beam_search") {
      beam_search_nodes.push_back(node);
    }
  }
  return beam_search_nodes.size() == 1 &&
         beam_search_nodes[0]->Op()->GetAttrIfExists<int>("beam_size") == 1;
}

void OneBeamSizeFusePass::RemoveAssignGather(ir::Graph* graph) const {
  // detect assign + gather
  GraphPatternDetector gpd;
  patterns::AssignPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle RemoveAssignGather";
    GET_IR_NODE(assign);
    GET_IR_NODE(assign_out);
    // Assign_out may not link to gather, so we find gather by input name.
    auto next_ops = FindOpNodeByInputName(graph, assign_out->Name());
    if (next_ops.size() != 1 || next_ops[0]->Name() != "gather") return;
    auto* gather = next_ops[0];

    // "assign_out" is used in multi blocks. "assign_out" should be reserved.
    auto* assign_in = assign->inputs[0];
    auto* fused_multi_transformer = assign_in->inputs[0];
    fused_multi_transformer->Op()->Rename(assign_in->Name(),
                                          assign_out->Name());
    IR_NODE_LINK_TO(fused_multi_transformer, assign_out);

    std::unordered_set<const Node*> delete_nodes{assign, assign_in, gather};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void OneBeamSizeFusePass::FoldShapeAssociatedOps(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::ShapeAssociatedOpsPattern pattern(gpd.mutable_pattern(),
                                              name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FoldShapeAssociatedOps";
    GET_IR_NODE(shape);
    GET_IR_NODE(slice);
    GET_IR_NODE(div);
    GET_IR_NODE(cast_0);
    GET_IR_NODE(cast_1);
    GET_IR_NODE(scale_0);
    GET_IR_NODE(cast_2);
    GET_IR_NODE(range);
    GET_IR_NODE(unsqueeze2);
    GET_IR_NODE(scale_1);
    GET_IR_NODE(add);
    GET_IR_NODE(flatten_contiguous_range);
    GET_IR_NODE(shape_out);
    GET_IR_NODE(slice_out);
    GET_IR_NODE(div_out);
    GET_IR_NODE(cast_0_out);
    GET_IR_NODE(cast_1_out);
    GET_IR_NODE(scale_0_out);
    GET_IR_NODE(cast_2_out);
    GET_IR_NODE(range_out);
    GET_IR_NODE(unsqueeze2_out);
    GET_IR_NODE(scale_1_out);
    GET_IR_NODE(add_x);
    GET_IR_NODE(add_out);

    flatten_contiguous_range->Op()->RenameInput(add_out->Name(), add_x->Name());

    std::unordered_set<const Node*> delete_nodes{
        shape,       slice,       div,        cast_0,     cast_1,
        scale_0,     cast_2,      range,      unsqueeze2, scale_1,
        add,         shape_out,   slice_out,  div_out,    cast_0_out,
        cast_1_out,  scale_0_out, cast_2_out, range_out,  unsqueeze2_out,
        scale_1_out, add_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void OneBeamSizeFusePass::RemoveBeamSearchAssociatedOps(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::BeamSearchAssociatedOpsPattern pattern(gpd.mutable_pattern(),
                                                   name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle RemoveBeamSearchAssociatedOps";
    GET_IR_NODE(lod_reset_0);
    GET_IR_NODE(lod_reset_1);
    GET_IR_NODE(beam_search);
    GET_IR_NODE(write_to_array_0);
    GET_IR_NODE(write_to_array_1);
    GET_IR_NODE(is_empty);
    GET_IR_NODE(logical_not);
    GET_IR_NODE(cast);
    GET_IR_NODE(lod_reset_0_out);
    GET_IR_NODE(lod_reset_1_out);
    GET_IR_NODE(beam_search_parent_idx);
    GET_IR_NODE(beam_search_selected_ids);
    GET_IR_NODE(beam_search_selected_scores);
    GET_IR_NODE(is_empty_out);
    GET_IR_NODE(logical_not_out);
    GET_IR_NODE(cast_out);

    auto* block = lod_reset_0->Op()->Block();
    auto* scope = param_scope();

    write_to_array_0->Op()->RenameInput(beam_search_selected_ids->Name(),
                                        lod_reset_0_out->Name());
    IR_NODE_LINK_TO(lod_reset_0_out, write_to_array_0);
    write_to_array_1->Op()->RenameInput(beam_search_selected_scores->Name(),
                                        lod_reset_1_out->Name());
    IR_NODE_LINK_TO(lod_reset_1_out, write_to_array_1);

    // Transform is_empty to not_equal
    is_empty->RenameOp("not_equal");
    auto* not_equal = is_empty;
    auto* not_equal_desc = not_equal->Op();
    not_equal_desc->RenameInput(beam_search_selected_ids->Name(),
                                lod_reset_0_out->Name());
    not_equal_desc->RenameOutput(is_empty_out->Name(), logical_not_out->Name());
    std::string not_equal_y_name = lod_reset_0_out->Name() + "_not_equal_y";
    not_equal_desc->SetInput("Y", {not_equal_y_name});

    VarDesc not_equal_y_desc(not_equal_y_name);
    not_equal_y_desc.SetPersistable(true);
    not_equal_y_desc.SetShape({static_cast<int64_t>(1)});
    not_equal_y_desc.SetDataType(proto::VarType::Type::VarType_Type_INT64);
    auto* not_equal_y = graph->CreateVarNode(&not_equal_y_desc);
    auto* block_not_equal_y_desc = block->Var(not_equal_y_name);
    block_not_equal_y_desc->SetPersistable(not_equal_y_desc.Persistable());
    block_not_equal_y_desc->SetShape(not_equal_y_desc.GetShape());
    block_not_equal_y_desc->SetDataType(not_equal_y_desc.GetDataType());
    auto* not_equal_y_tensor =
        scope->Var(not_equal_y_name)->GetMutable<phi::DenseTensor>();
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    not_equal_y_tensor->Resize({1});
    not_equal_y_tensor->set_type(phi::DataType::INT64);
    auto* not_equal_y_data = cpu_ctx->Alloc<int64_t>(not_equal_y_tensor);
    not_equal_y_data[0] = beam_search->Op()->GetAttrIfExists<int>("end_id");
    IR_NODE_LINK_TO(not_equal_y, not_equal);

    // cast_out is 0
    cast_out->Var()->SetPersistable(true);
    auto* cast_out_tensor =
        scope->Var(cast_out->Name())->GetMutable<phi::DenseTensor>();
    cast_out_tensor->Resize({1});
    cast_out_tensor->set_type(phi::DataType::INT64);
    auto* cast_out_data = cpu_ctx->Alloc<int64_t>(cast_out_tensor);
    cast_out_data[0] = 0;

    std::unordered_set<const Node*> delete_nodes{
        beam_search,
        logical_not,
        cast,
        beam_search_parent_idx,
        beam_search_selected_ids,
        beam_search_selected_scores,
        is_empty_out,
    };
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

namespace patterns {
struct WriteToArrayPattern : public PatternBase {
  WriteToArrayPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(write_to_array);
  // declare variable node's name
  PATTERN_DECL_NODE(write_x);
  PATTERN_DECL_NODE(write_out);
};

WriteToArrayPattern::WriteToArrayPattern(PDPattern* pattern,
                                         const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* write_x = pattern->NewNode(write_x_repr())
                      ->assert_is_op_input("write_to_array", "X")
                      ->assert_is_persistable_var();
  auto* write_to_array =
      pattern->NewNode(write_to_array_repr())->assert_is_op("write_to_array");
  auto* write_out = pattern->NewNode(write_out_repr())
                        ->assert_is_op_output("write_to_array", "Out");

  write_to_array->LinksFrom({write_x}).LinksTo({write_out});
}
}  // namespace patterns

void OneBeamSizeFusePass::RemoveWriteReadArrayOps(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::WriteToArrayPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle RemoveWriteReadArrayOps";
    GET_IR_NODE(write_to_array);
    GET_IR_NODE(write_x);
    GET_IR_NODE(write_out);
    auto* scope = param_scope();

    // write_out is from graph0 and do not link to any op, so we find
    // "read_from_array" by write_out name.
    auto next_ops = FindOpNodeByInputName(graph, write_out->Name());
    if (next_ops.size() != 1 || next_ops[0]->Name() != "read_from_array")
      return;
    auto* read_from_array = next_ops[0];
    auto* read_out = read_from_array->outputs[0];
    read_out->Var()->SetPersistable(true);
    auto* write_x_tensor =
        scope->Var(write_x->Name())->GetMutable<phi::DenseTensor>();
    auto* read_out_tensor =
        scope->Var(read_out->Name())->GetMutable<phi::DenseTensor>();
    Assign(*write_x_tensor, read_out_tensor);

    std::unordered_set<const Node*> delete_nodes{
        write_to_array, write_out, read_from_array};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

namespace patterns {
struct GatherPattern : public PatternBase {
  GatherPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(gather);
  // declare variable node's name
  PATTERN_DECL_NODE(gather_x);
  PATTERN_DECL_NODE(gather_i);
  PATTERN_DECL_NODE(gather_out);
};

GatherPattern::GatherPattern(PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* gather_x =
      pattern->NewNode(gather_x_repr())->assert_is_op_input("gather", "X");
  auto* gather_i = pattern->NewNode(gather_i_repr())
                       ->assert_is_op_input("gather", "Index")
                       ->assert_is_persistable_var();
  auto* gather = pattern->NewNode(gather_repr())
                     ->assert_is_op("gather")
                     ->assert_more([&](Node* node) {
                       return node->Op()->GetAttrIfExists<int>("axis") == 0;
                     });
  auto* gather_out =
      pattern->NewNode(gather_out_repr())->assert_is_op_output("gather", "Out");

  gather->LinksFrom({gather_x, gather_i}).LinksTo({gather_out});
}
}  // namespace patterns

void OneBeamSizeFusePass::RemoveGatherOps(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::GatherPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle RemoveGatherOps";
    GET_IR_NODE(gather);
    GET_IR_NODE(gather_x);
    GET_IR_NODE(gather_i);
    GET_IR_NODE(gather_out);
    auto* scope = param_scope();

    // gather_i should be 0
    auto* gather_i_tensor =
        scope->Var(gather_i->Name())->GetMutable<phi::DenseTensor>();
    auto gather_i_dims = gather_i_tensor->dims();
    if (gather_i_dims.size() != 1 || gather_i_dims[0] != 1) return;
    if (gather_i_tensor->dtype() == phi::DataType::INT32) {
      auto* i_data = gather_i_tensor->data<int>();
      if (i_data[0] != 0) return;
    } else {
      auto* i_data = gather_i_tensor->data<int64_t>();
      if (i_data[0] != 0) return;
    }

    auto gather_x_name = gather_x->Name();
    auto gather_out_name = gather_out->Name();
    for (auto* next_op : gather_out->outputs) {
      next_op->Op()->RenameInput(gather_out_name, gather_x_name);
      IR_NODE_LINK_TO(gather_x, next_op);
    }

    std::unordered_set<const Node*> delete_nodes{gather, gather_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void OneBeamSizeFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  if (!OnlyOneBeamSearchAndOneBeamSize(graph)) return;
  RemoveAssignGather(graph);
  FoldShapeAssociatedOps(graph);
  RemoveBeamSearchAssociatedOps(graph);
  RemoveWriteReadArrayOps(graph);
  RemoveGatherOps(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(one_beam_size_fuse_pass,
              paddle::framework::ir::OneBeamSizeFusePass);

REGISTER_PASS_CAPABILITY(one_beam_size_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "beam_search", 0));
