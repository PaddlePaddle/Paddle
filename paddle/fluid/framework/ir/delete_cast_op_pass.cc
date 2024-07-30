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

#include "paddle/fluid/framework/ir/delete_cast_op_pass.h"

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/kernels/assign_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle::framework {
class Scope;
}  // namespace paddle::framework

namespace paddle::framework::ir::patterns {
struct CastWritePattern : public PatternBase {
  CastWritePattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(cast0);
  PATTERN_DECL_NODE(write_to_array);
  // declare variable node's name
  PATTERN_DECL_NODE(cast0_in);
  PATTERN_DECL_NODE(cast0_out);
  PATTERN_DECL_NODE(write_to_array_out);
};

CastWritePattern::CastWritePattern(PDPattern* pattern,
                                   const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* cast0_in =
      pattern->NewNode(cast0_in_repr())->assert_is_op_input("cast", "X");
  auto* cast0 =
      pattern->NewNode(cast0_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP16) &&
                   out_dtype == static_cast<int>(proto::VarType::FP32);
          });
  auto* cast0_out = pattern->NewNode(cast0_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("write_to_array", "X")
                        ->assert_has_n_outputs(1);
  auto* write_to_array =
      pattern->NewNode(write_to_array_repr())->assert_is_op("write_to_array");
  auto* write_to_array_out = pattern->NewNode(write_to_array_out_repr())
                                 ->assert_is_op_output("write_to_array", "Out");

  cast0->LinksFrom({cast0_in}).LinksTo({cast0_out});
  write_to_array->LinksFrom({cast0_out}).LinksTo({write_to_array_out});
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

static std::vector<Node*> FindOpNodeWithInputName(
    ir::Graph* graph, const std::string& input_name) {
  std::vector<Node*> ret;
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) continue;
    auto inputs = node->Op()->Inputs();
    bool find_input = false;
    for (auto const& input : inputs) {
      auto input_names = input.second;
      if (std::count(input_names.begin(), input_names.end(), input_name) > 0) {
        find_input = true;
        break;
      }
    }
    if (find_input) ret.push_back(node);
  }
  return ret;
}

static std::vector<Node*> FindOpNodeWithOutputName(
    ir::Graph* graph, const std::string& output_name) {
  std::vector<Node*> ret;
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) continue;
    auto outputs = node->Op()->Outputs();
    bool find_output = false;
    for (auto const& output : outputs) {
      auto output_names = output.second;
      if (std::count(output_names.begin(), output_names.end(), output_name) >
          0) {
        find_output = true;
        break;
      }
    }
    if (find_output) ret.push_back(node);
  }
  return ret;
}

int DeleteCastOpPass::ApplyCastWriteReadPass(ir::Graph* graph) const {
  if (graph->SubGraphsSize() != 2) {
    VLOG(3) << "ApplyCastWriteReadPass only support 2 subgraphs.";
    return 0;
  }
  auto* graph0 = graph->GetSubGraph(0);
  auto* graph1 = graph->GetSubGraph(1);
  GraphPatternDetector gpd;
  patterns::CastWritePattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastWriteReadPass fuse";
    GET_IR_NODE_FROM_SUBGRAPH(cast0, cast0, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(write_to_array, write_to_array, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_in, cast0_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_out, cast0_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(write_to_array_out, write_to_array_out, pattern);

    // write_to_array_out(in graph1) may not link to any op nodes, so we fine
    // read_from_array by write_to_array_out name.
    auto write_out_op_nodes =
        FindOpNodeWithInputName(graph, write_to_array_out->Name());
    if (write_out_op_nodes.size() != 1 ||
        write_out_op_nodes[0]->Op()->Type() != "read_from_array")
      return;
    Node* read_from_array = write_out_op_nodes[0];
    Node* read_from_array_out = read_from_array->outputs[0];
    auto read_out_op_nodes =
        FindOpNodeWithInputName(graph, read_from_array_out->Name());
    if (read_out_op_nodes.size() != 1 ||
        read_out_op_nodes[0]->Op()->Type() != "cast")
      return;
    Node* cast1 = read_out_op_nodes[0];
    Node* cast1_out = cast1->outputs[0];

    // find nodes in graph0
    auto nodes_in_graph0 =
        FindOpNodeWithOutputName(graph0, write_to_array_out->Name());
    if (nodes_in_graph0.size() != 2) return;
    Node* write_to_array_0 = nullptr;
    Node* while_op = nullptr;
    for (auto* node : nodes_in_graph0) {
      if (node->Name() == "write_to_array") {
        write_to_array_0 = node;
      } else if (node->Name() == "while") {
        while_op = node;
      }
    }
    if (write_to_array_0 == nullptr || while_op == nullptr) return;

    // modify graph0
    Node* write_to_array_0_x = nullptr;
    auto write_to_array_0_x_name = write_to_array_0->Op()->Input("X")[0];
    for (auto* node : write_to_array_0->inputs) {
      if (node->Name() == write_to_array_0_x_name) {
        write_to_array_0_x = node;
        break;
      }
    }

    std::string cast_out_name = write_to_array_0_x_name + "_fp16";
    VarDesc cast_out_desc(cast_out_name);
    cast_out_desc.SetShape(write_to_array_0_x->Var()->GetShape());  // NOLINT
    cast_out_desc.SetDataType(proto::VarType::Type::VarType_Type_FP16);
    auto* cast_out = graph0->CreateVarNode(&cast_out_desc);

    auto* block = write_to_array_0->Op()->Block();
    framework::OpDesc cast_op_desc(block);
    cast_op_desc.SetType("cast");
    cast_op_desc.SetInput("X", {write_to_array_0_x_name});
    cast_op_desc.SetAttr("in_dtype", 5);
    cast_op_desc.SetAttr("out_dtype", 4);
    cast_op_desc.SetOutput("Out", {cast_out_name});
    auto* cast = graph0->CreateOpNode(&cast_op_desc);

    write_to_array_0->Op()->RenameInput(write_to_array_0_x_name, cast_out_name);

    IR_NODE_UNLINK(write_to_array_0_x, write_to_array_0);
    IR_NODE_LINK_TO(write_to_array_0_x, cast);
    IR_NODE_LINK_TO(cast, cast_out);
    IR_NODE_LINK_TO(cast_out, write_to_array_0);

    // modify graph1
    write_to_array->Op()->RenameInput(cast0_out->Name(), cast0_in->Name());
    read_from_array->Op()->RenameOutput(read_from_array_out->Name(),
                                        cast1_out->Name());
    IR_NODE_LINK_TO(cast0, write_to_array);
    IR_NODE_LINK_TO(read_from_array_out, cast1_out);

    std::unordered_set<const Node*> delete_nodes{
        cast0, cast1, cast0_out, read_from_array_out};
    GraphSafeRemoveNodes(graph, delete_nodes);

    found_subgraph_count++;
  };

  gpd(graph1, handler);
  return found_subgraph_count;
}

}  // namespace paddle::framework::ir
namespace paddle::framework::ir::patterns {
struct CastLodResetWritePattern : public PatternBase {
  CastLodResetWritePattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(cast0);
  PATTERN_DECL_NODE(lod_reset);
  PATTERN_DECL_NODE(write_to_array);
  // declare variable node's name
  PATTERN_DECL_NODE(cast0_in);
  PATTERN_DECL_NODE(cast0_out);
  PATTERN_DECL_NODE(lod_reset_out);
  PATTERN_DECL_NODE(write_to_array_out);
};

CastLodResetWritePattern::CastLodResetWritePattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* cast0_in =
      pattern->NewNode(cast0_in_repr())->assert_is_op_input("cast", "X");
  auto* cast0 =
      pattern->NewNode(cast0_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP16) &&
                   out_dtype == static_cast<int>(proto::VarType::FP32);
          });
  auto* cast0_out = pattern->NewNode(cast0_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("lod_reset", "X")
                        ->assert_has_n_outputs(1);
  auto* lod_reset =
      pattern->NewNode(lod_reset_repr())->assert_is_op("lod_reset");
  auto* lod_reset_out = pattern->NewNode(lod_reset_out_repr())
                            ->assert_is_op_output("lod_reset", "Out")
                            ->assert_is_op_input("write_to_array", "X")
                            ->assert_has_n_outputs(1);
  auto* write_to_array =
      pattern->NewNode(write_to_array_repr())->assert_is_op("write_to_array");
  auto* write_to_array_out = pattern->NewNode(write_to_array_out_repr())
                                 ->assert_is_op_output("write_to_array", "Out");

  cast0->LinksFrom({cast0_in}).LinksTo({cast0_out});
  lod_reset->LinksFrom({cast0_out}).LinksTo({lod_reset_out});
  write_to_array->LinksFrom({lod_reset_out}).LinksTo({write_to_array_out});
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

int DeleteCastOpPass::ApplyCastLodResetWriteReadPass(ir::Graph* graph) const {
  if (graph->SubGraphsSize() != 2) {
    VLOG(3) << "ApplyCastLodResetWriteReadPass only support 2 subgraphs.";
    return 0;
  }
  auto* graph0 = graph->GetSubGraph(0);
  auto* graph1 = graph->GetSubGraph(1);
  GraphPatternDetector gpd;
  patterns::CastLodResetWritePattern pattern(gpd.mutable_pattern(),
                                             name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastLodResetWriteReadPass fuse";
    GET_IR_NODE_FROM_SUBGRAPH(cast0, cast0, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lod_reset, lod_reset, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(write_to_array, write_to_array, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_in, cast0_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_out, cast0_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lod_reset_out, lod_reset_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(write_to_array_out, write_to_array_out, pattern);

    // write_to_array_out(in graph1) may not link to any op nodes, so we fine
    // read_from_array by write_to_array_out name.
    auto write_out_op_nodes =
        FindOpNodeWithInputName(graph, write_to_array_out->Name());
    if (write_out_op_nodes.size() != 1 ||
        write_out_op_nodes[0]->Op()->Type() != "read_from_array")
      return;
    Node* read_from_array = write_out_op_nodes[0];
    Node* read_from_array_out = read_from_array->outputs[0];
    auto read_out_op_nodes =
        FindOpNodeWithInputName(graph, read_from_array_out->Name());
    if (read_out_op_nodes.size() != 1 ||
        read_out_op_nodes[0]->Op()->Type() != "cast")
      return;
    Node* cast1 = read_out_op_nodes[0];
    Node* cast1_out = cast1->outputs[0];

    // find nodes in graph0
    auto nodes_in_graph0 =
        FindOpNodeWithOutputName(graph0, write_to_array_out->Name());
    if (nodes_in_graph0.size() != 2) return;
    Node* write_to_array_0 = nullptr;
    Node* while_op = nullptr;
    for (auto* node : nodes_in_graph0) {
      if (node->Name() == "write_to_array") {
        write_to_array_0 = node;
      } else if (node->Name() == "while") {
        while_op = node;
      }
    }
    if (write_to_array_0 == nullptr || while_op == nullptr) return;

    nodes_in_graph0 =
        FindOpNodeWithInputName(graph0, write_to_array_out->Name());
    if (nodes_in_graph0.size() != 2) return;
    Node* beam_search_decode = nullptr;
    while_op = nullptr;
    for (auto* node : nodes_in_graph0) {
      if (node->Name() == "beam_search_decode") {
        beam_search_decode = node;
      } else if (node->Name() == "while") {
        while_op = node;
      }
    }
    if (beam_search_decode == nullptr || while_op == nullptr) return;

    // modify graph0: 1. insert cast before write_to_array_0
    Node* write_to_array_0_x = nullptr;
    auto write_to_array_0_x_name = write_to_array_0->Op()->Input("X")[0];
    for (auto* node : write_to_array_0->inputs) {
      if (node->Name() == write_to_array_0_x_name) {
        write_to_array_0_x = node;
        break;
      }
    }

    std::string cast_out_name = write_to_array_0_x_name + "_fp16";
    VarDesc cast_out_desc(cast_out_name);
    cast_out_desc.SetShape(write_to_array_0_x->Var()->GetShape());
    cast_out_desc.SetDataType(proto::VarType::Type::VarType_Type_FP16);
    auto* cast_out = graph0->CreateVarNode(&cast_out_desc);

    auto* block = write_to_array_0->Op()->Block();
    framework::OpDesc cast_op_desc(block);
    cast_op_desc.SetType("cast");
    cast_op_desc.SetInput("X", {write_to_array_0_x_name});
    cast_op_desc.SetAttr("in_dtype", 5);
    cast_op_desc.SetAttr("out_dtype", 4);
    cast_op_desc.SetOutput("Out", {cast_out_name});
    auto* cast = graph0->CreateOpNode(&cast_op_desc);

    write_to_array_0->Op()->RenameInput(write_to_array_0_x_name, cast_out_name);
    IR_NODE_UNLINK(write_to_array_0_x, write_to_array_0);
    IR_NODE_LINK_TO(write_to_array_0_x, cast);
    IR_NODE_LINK_TO(cast, cast_out);
    IR_NODE_LINK_TO(cast_out, write_to_array_0);

    // modify graph0: 2. insert cast after beam_search_decode
    Node* beam_search_decode_out_score = nullptr;
    for (auto* node : beam_search_decode->outputs) {
      if (node->Name() ==
          beam_search_decode->Op()->Output("SentenceScores")[0]) {
        beam_search_decode_out_score = node;
        break;
      }
    }

    std::string cast_in_name = beam_search_decode_out_score->Name() + "_fp16";
    VarDesc cast_in_desc(cast_in_name);
    cast_in_desc.SetShape(beam_search_decode_out_score->Var()->GetShape());
    cast_in_desc.SetDataType(proto::VarType::Type::VarType_Type_FP16);
    auto* cast_in = graph0->CreateVarNode(&cast_in_desc);

    cast_op_desc = framework::OpDesc(block);
    cast_op_desc.SetType("cast");
    cast_op_desc.SetInput("X", {cast_in_name});
    cast_op_desc.SetAttr("in_dtype", 4);
    cast_op_desc.SetAttr("out_dtype", 5);
    cast_op_desc.SetOutput("Out", {beam_search_decode_out_score->Name()});
    cast = graph0->CreateOpNode(&cast_op_desc);

    beam_search_decode->Op()->RenameOutput(beam_search_decode_out_score->Name(),
                                           cast_in_name);
    IR_NODE_UNLINK(beam_search_decode, beam_search_decode_out_score);
    IR_NODE_LINK_TO(beam_search_decode, cast_in);
    IR_NODE_LINK_TO(cast_in, cast);
    IR_NODE_LINK_TO(cast, beam_search_decode_out_score);

    // modify graph1
    lod_reset->Op()->RenameInput(cast0_out->Name(), cast0_in->Name());
    read_from_array->Op()->RenameOutput(read_from_array_out->Name(),
                                        cast1_out->Name());
    IR_NODE_LINK_TO(cast0, lod_reset);
    IR_NODE_LINK_TO(read_from_array_out, cast1_out);

    std::unordered_set<const Node*> delete_nodes{
        cast0, cast1, cast0_out, read_from_array_out};
    GraphSafeRemoveNodes(graph, delete_nodes);

    found_subgraph_count++;
  };

  gpd(graph1, handler);
  return found_subgraph_count;
}

}  // namespace paddle::framework::ir
namespace paddle::framework::ir::patterns {
struct CastIndexSamplePattern : public PatternBase {
  CastIndexSamplePattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(cast0);
  PATTERN_DECL_NODE(index_sample);
  PATTERN_DECL_NODE(cast1);
  // declare variable node's name
  PATTERN_DECL_NODE(cast0_in);
  PATTERN_DECL_NODE(cast0_out);
  PATTERN_DECL_NODE(index_sample_out);
  PATTERN_DECL_NODE(cast1_out);
};

CastIndexSamplePattern::CastIndexSamplePattern(PDPattern* pattern,
                                               const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* cast0_in =
      pattern->NewNode(cast0_in_repr())->assert_is_op_input("cast", "X");
  auto* cast0 =
      pattern->NewNode(cast0_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP16) &&
                   out_dtype == static_cast<int>(proto::VarType::FP32);
          });
  auto* cast0_out = pattern->NewNode(cast0_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("index_sample", "X")
                        ->assert_has_n_outputs(1);
  auto* index_sample =
      pattern->NewNode(index_sample_repr())->assert_is_op("index_sample");
  auto* index_sample_out = pattern->NewNode(index_sample_out_repr())
                               ->assert_is_op_output("index_sample", "Out")
                               ->assert_is_op_input("cast", "X")
                               ->assert_has_n_outputs(1);
  auto* cast1 =
      pattern->NewNode(cast1_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP32) &&
                   out_dtype == static_cast<int>(proto::VarType::FP16);
          });
  auto* cast1_out =
      pattern->NewNode(cast1_out_repr())->assert_is_op_output("cast", "Out");

  cast0->LinksFrom({cast0_in}).LinksTo({cast0_out});
  index_sample->LinksFrom({cast0_out}).LinksTo({index_sample_out});
  cast1->LinksFrom({index_sample_out}).LinksTo({cast1_out});
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

int DeleteCastOpPass::ApplyCastIndexSamplePass(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::CastIndexSamplePattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastIndexSamplePass fuse";
    GET_IR_NODE_FROM_SUBGRAPH(cast0, cast0, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(index_sample, index_sample, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1, cast1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_in, cast0_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_out, cast0_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(index_sample_out, index_sample_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1_out, cast1_out, pattern);

    index_sample->Op()->RenameInput(cast0_out->Name(), cast0_in->Name());
    index_sample->Op()->RenameOutput(index_sample_out->Name(),
                                     cast1_out->Name());
    IR_NODE_LINK_TO(cast0_in, index_sample);
    IR_NODE_LINK_TO(index_sample, cast1_out);

    std::unordered_set<const Node*> delete_nodes{
        cast0, cast1, cast0_out, index_sample_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace paddle::framework::ir
namespace paddle::framework::ir::patterns {
struct CastScatterPattern : public PatternBase {
  CastScatterPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(scatter);
  PATTERN_DECL_NODE(cast0);
  PATTERN_DECL_NODE(cast1);
  PATTERN_DECL_NODE(cast2);
  // declare variable node's name
  PATTERN_DECL_NODE(cast0_in);
  PATTERN_DECL_NODE(cast0_out);
  PATTERN_DECL_NODE(cast1_in);
  PATTERN_DECL_NODE(cast1_out);
  PATTERN_DECL_NODE(scatter_out);
  PATTERN_DECL_NODE(cast2_out);
};

CastScatterPattern::CastScatterPattern(PDPattern* pattern,
                                       const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* cast0_in = pattern->NewNode(cast0_in_repr())
                       ->assert_is_op_input("cast", "X")
                       ->assert_has_n_outputs(1);
  auto* cast0 =
      pattern->NewNode(cast0_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP16) &&
                   out_dtype == static_cast<int>(proto::VarType::FP32);
          });
  auto* cast0_out = pattern->NewNode(cast0_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("scatter", "X")
                        ->assert_has_n_outputs(1);
  auto* cast1_in = pattern->NewNode(cast1_in_repr())
                       ->assert_is_op_input("cast", "X")
                       ->assert_has_n_outputs(1);
  auto* cast1 =
      pattern->NewNode(cast1_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP16) &&
                   out_dtype == static_cast<int>(proto::VarType::FP32);
          });
  auto* cast1_out = pattern->NewNode(cast1_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("scatter", "Updates")
                        ->assert_has_n_outputs(1);
  auto* scatter = pattern->NewNode(scatter_repr())->assert_is_op("scatter");
  auto* scatter_out = pattern->NewNode(scatter_out_repr())
                          ->assert_is_op_output("scatter", "Out")
                          ->assert_is_op_input("cast", "X")
                          ->assert_has_n_outputs(1);
  auto* cast2 =
      pattern->NewNode(cast2_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP32) &&
                   out_dtype == static_cast<int>(proto::VarType::FP16);
          });
  auto* cast2_out =
      pattern->NewNode(cast2_out_repr())->assert_is_op_output("cast", "Out");

  cast0->LinksFrom({cast0_in}).LinksTo({cast0_out});
  cast1->LinksFrom({cast1_in}).LinksTo({cast1_out});
  scatter->LinksFrom({cast0_out, cast1_out}).LinksTo({scatter_out});
  cast2->LinksFrom({scatter_out}).LinksTo({cast2_out});
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

int DeleteCastOpPass::ApplyCastScatterPass(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::CastScatterPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastScatterPass fuse";
    GET_IR_NODE_FROM_SUBGRAPH(scatter, scatter, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0, cast0, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1, cast1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast2, cast2, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_in, cast0_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast0_out, cast0_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1_in, cast1_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast1_out, cast1_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scatter_out, scatter_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast2_out, cast2_out, pattern);

    scatter->Op()->RenameInput(cast0_out->Name(), cast0_in->Name());
    scatter->Op()->RenameInput(cast1_out->Name(), cast1_in->Name());
    scatter->Op()->RenameOutput(scatter_out->Name(), cast2_out->Name());
    IR_NODE_LINK_TO(cast0_in, scatter);
    IR_NODE_LINK_TO(cast1_in, scatter);
    IR_NODE_LINK_TO(scatter, cast2_out);

    std::unordered_set<const Node*> delete_nodes{
        cast0, cast1, cast2, cast0_out, cast1_out, scatter_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace paddle::framework::ir
namespace paddle::framework::ir::patterns {
struct CastLookupTablePattern : public PatternBase {
  CastLookupTablePattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(lookup_table);
  PATTERN_DECL_NODE(cast);
  // declare variable node's name
  PATTERN_DECL_NODE(lookup_table_w);
  PATTERN_DECL_NODE(lookup_table_out);
  PATTERN_DECL_NODE(cast_out);
};

CastLookupTablePattern::CastLookupTablePattern(PDPattern* pattern,
                                               const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* lookup_table_w = pattern->NewNode(lookup_table_w_repr())
                             ->assert_is_op_input("lookup_table_v2", "W")
                             ->assert_is_persistable_var();
  auto* lookup_table =
      pattern->NewNode(lookup_table_repr())->assert_is_op("lookup_table_v2");
  auto* lookup_table_out = pattern->NewNode(lookup_table_out_repr())
                               ->assert_is_op_output("lookup_table_v2", "Out")
                               ->assert_is_op_input("cast", "X")
                               ->assert_has_n_outputs(1);
  auto* cast =
      pattern->NewNode(cast_repr())
          ->assert_is_op("cast")
          ->assert_more([](Node* node) {
            auto* op_desc = node->Op();
            auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
            auto out_dtype = op_desc->GetAttrIfExists<int>("out_dtype");
            return in_dtype == static_cast<int>(proto::VarType::FP32) &&
                   out_dtype == static_cast<int>(proto::VarType::FP16);
          });
  auto* cast_out =
      pattern->NewNode(cast_out_repr())->assert_is_op_output("cast", "Out");

  lookup_table->LinksFrom({lookup_table_w}).LinksTo({lookup_table_out});
  cast->LinksFrom({lookup_table_out}).LinksTo({cast_out});
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

int DeleteCastOpPass::ApplyCastLookupTablePass(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::CastLookupTablePattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastLookupTablePass fuse";
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table, lookup_table, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast, cast, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table_w, lookup_table_w, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(lookup_table_out, lookup_table_out, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast_out, cast_out, pattern);
    auto* scope = param_scope();

    auto* w_tensor =
        scope->Var(lookup_table_w->Name())->GetMutable<phi::DenseTensor>();
    lookup_table_w->Var()->SetDataType(proto::VarType::FP16);
    if (w_tensor->dtype() != phi::DataType::FLOAT16) {
      auto* cpu_ctx = static_cast<phi::CPUContext*>(
          phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
      phi::DenseTensor w_fp32_tensor;
      w_fp32_tensor.Resize(w_tensor->dims());
      w_fp32_tensor.set_type(w_tensor->dtype());
      phi::AssignKernel(*cpu_ctx, *w_tensor, &w_fp32_tensor);
      w_tensor->set_type(phi::DataType::FLOAT16);
      phi::CastKernel<float>(
          *cpu_ctx, w_fp32_tensor, phi::DataType::FLOAT16, w_tensor);
    }

    for (auto* next_op : cast_out->outputs) {
      next_op->Op()->RenameInput(cast_out->Name(), lookup_table_out->Name());
      IR_NODE_LINK_TO(lookup_table_out, next_op);
    }

    std::unordered_set<const Node*> delete_nodes{cast, cast_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace paddle::framework::ir
namespace paddle::framework::ir::patterns {
struct CastPattern : public PatternBase {
  CastPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(cast);
  // declare variable node's name
  PATTERN_DECL_NODE(cast_in);
  PATTERN_DECL_NODE(cast_out);
};

CastPattern::CastPattern(PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* cast_in =
      pattern->NewNode(cast_in_repr())->assert_is_op_input("cast", "X");
  auto* cast = pattern->NewNode(cast_repr())
                   ->assert_is_op("cast")
                   ->assert_more([](Node* node) {
                     auto* op_desc = node->Op();
                     auto in_dtype = op_desc->GetAttrIfExists<int>("in_dtype");
                     auto out_dtype =
                         op_desc->GetAttrIfExists<int>("out_dtype");
                     return in_dtype == out_dtype;
                   });
  auto* cast_out =
      pattern->NewNode(cast_out_repr())->assert_is_op_output("cast", "Out");

  cast->LinksFrom({cast_in}).LinksTo({cast_out});
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

int DeleteCastOpPass::ApplyCastPass(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::CastPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyCastPass fuse";
    GET_IR_NODE_FROM_SUBGRAPH(cast, cast, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast_in, cast_in, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(cast_out, cast_out, pattern);
    for (auto* out_op_node : cast_out->outputs) {
      out_op_node->Op()->RenameInput(cast_out->Name(), cast_in->Name());
      IR_NODE_LINK_TO(cast_in, out_op_node);
    }
    std::unordered_set<const Node*> delete_nodes{cast, cast_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void DeleteCastOpPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  if (!graph->IsMainGraph()) {
    VLOG(3) << "'delete_cast_op_pass' needs info in all graphs, so it "
               "should be applied in the main graph.";
    return;
  }
  Init(name_scope_, graph);

  int found_subgraph_count = ApplyCastWriteReadPass(graph);
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " cast_write_read_cast subgraph";
  }

  found_subgraph_count = ApplyCastLodResetWriteReadPass(graph);
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " cast_lod_reset_write_read_cast subgraph";
  }

  found_subgraph_count = 0;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    found_subgraph_count += ApplyCastIndexSamplePass(graph->GetSubGraph(i));
  }
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " cast_index_sample_cast subgraph";
  }

  found_subgraph_count = 0;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    found_subgraph_count += ApplyCastScatterPass(graph->GetSubGraph(i));
  }
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " cast_scatter_cast subgraph";
  }

  found_subgraph_count = 0;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    found_subgraph_count += ApplyCastLookupTablePass(graph->GetSubGraph(i));
  }
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " lookup_table_cast subgraph";
  }

  found_subgraph_count = 0;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    found_subgraph_count += ApplyCastPass(graph->GetSubGraph(i));
  }
  if (found_subgraph_count > 0) {
    LOG(INFO) << "--- delete " << found_subgraph_count
              << " cast(with same in/out dtype) subgraph";
  }
}

}  // namespace paddle::framework::ir

REGISTER_PASS(delete_cast_op_pass, paddle::framework::ir::DeleteCastOpPass);

REGISTER_PASS_CAPABILITY(delete_cast_op_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "cast", 0));
