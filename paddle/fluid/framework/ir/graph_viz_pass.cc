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

#include <algorithm>
#include <unordered_set>

#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/ir/subblock_to_graph_pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/inference/analysis/dot.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace framework {
namespace ir {
using inference::analysis::Dot;
namespace {
const char kGraphVizPath[] = "graph_viz_path";

std::string FormatName(const Node* node) {
  if (!node->IsOp() || !node->Op() ||
      !node->Op()->HasAttr(OpProtoAndCheckerMaker::OpNamescopeAttrName())) {
    return node->Name();
  }
  const std::string full_scope = boost::get<std::string>(
      node->Op()->GetAttr(OpProtoAndCheckerMaker::OpNamescopeAttrName()));
  return string::Sprintf("%s%s", full_scope.c_str(), node->Name().c_str());
}
}  // namespace

std::unique_ptr<ir::Graph> GraphVizPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  const std::string graph_viz_path = Get<std::string>(kGraphVizPath);
  VLOG(3) << "draw IR graph viz to " << graph_viz_path;
  std::unique_ptr<std::ostream> fout(new std::ofstream(graph_viz_path));
  PADDLE_ENFORCE(fout->good());
  std::ostream& sout = *fout;

  Dot dot;
  auto marked_nodes = ConsumeMarkedNodes(graph.get());
  DotDrawGraph(*graph, &dot, 0, marked_nodes);

  sout << dot.Build();

  return graph;
}

using subgraphs_t = SubblockToGraphPass::subgraphs_t;

void DotDrawGraph(const Graph& graph, paddle::inference::analysis::Dot* dot,
                  int node_id_offset,
                  const std::unordered_set<const Node*>& marked_nodes) {
  std::unordered_map<const ir::Node*, std::string> node2dot;

  const std::vector<Dot::Attr> op_attrs({
      Dot::Attr("style", "rounded,filled,bold"),  //
      Dot::Attr("shape", "box"),                  //
      Dot::Attr("color", "#303A3A"),              //
      Dot::Attr("fontcolor", "#ffffff"),          //
      Dot::Attr("width", "1.3"),                  //
      Dot::Attr("height", "0.84"),                //
      Dot::Attr("fontname", "Arial"),             //
  });
  const std::vector<Dot::Attr> arg_attrs({
      Dot::Attr("shape", "box"),                  //
      Dot::Attr("style", "rounded,filled,bold"),  //
      Dot::Attr("fontname", "Arial"),             //
      Dot::Attr("fillcolor", "#999999"),          //
      Dot::Attr("color", "#dddddd"),              //
  });

  const std::vector<Dot::Attr> param_attrs({
      Dot::Attr("shape", "box"),                  //
      Dot::Attr("style", "rounded,filled,bold"),  //
      Dot::Attr("fontname", "Arial"),             //
      Dot::Attr("color", "#148b97"),              //
      Dot::Attr("fontcolor", "#ffffff"),          //
  });

  const std::vector<Dot::Attr> marked_op_attrs(
      {Dot::Attr("style", "rounded,filled,bold"), Dot::Attr("shape", "box"),
       Dot::Attr("fillcolor", "yellow")});
  const std::vector<Dot::Attr> marked_var_attrs(
      {Dot::Attr("style", "filled,rounded"), Dot::Attr("shape", "box"),
       Dot::Attr("fillcolor", "yellow")});

  const std::vector<Dot::Attr> sub_graph_attrs(
      {Dot::Attr("style", "filled,rounded,dotted"),
       Dot::Attr("fillcolor", "black"), Dot::Attr("fillcolor", "azure"),
       Dot::Attr("color", "black"), Dot::Attr("border", "3"),
       Dot::Attr("margin", "10")});

  // try to get the sub graphs.
  subgraphs_t* sub_graphs{nullptr};
  if (graph.Has(kSubblockGraphAttr)) {
    sub_graphs =
        &graph.Get<SubblockToGraphPass::subgraphs_t>(kSubblockGraphAttr);
  }

  int num_sub_graph{0};
  auto node_num = static_cast<int>(graph.Nodes().size());

  // Create nodes
  for (const Node* n : graph.Nodes()) {
    std::string node_id =
        FormatName(n) + "(" + std::to_string(n->id() + node_id_offset) + ")";
    if (n->IsOp() && n->Op()) {
      decltype(op_attrs) attr =
          marked_nodes.count(n) ? marked_op_attrs : op_attrs;
      // the node_id is unique within a graph.
      dot->AddNode(node_id, attr, node_id);

      // Draw the sub-graph.
      if (sub_graphs) {
        auto it = sub_graphs->find(n);
        if (it != sub_graphs->end()) {
          Dot sub_graph_dot(sub_graph_attrs);
          DotDrawGraph(*it->second, &sub_graph_dot, node_num);
          node_num += it->second->Nodes().size();
          dot->AddSubgraph(sub_graph_dot.Build(
              "subgraph", "cluster_" + std::to_string(num_sub_graph++)));
          auto this_node_id = dot->GetNode(node_id).id();
          std::string one_node_in_sub_graph =
              sub_graph_dot.nodes().front().second.id();
          dot->AddRawEdge(
              this_node_id, one_node_in_sub_graph,
              std::vector<Dot::Attr>({Dot::Attr("ltail", this_node_id),
                                      Dot::Attr("lhead", one_node_in_sub_graph),
                                      Dot::Attr("color", "blue"),
                                      Dot::Attr("weight", "2")}));
        }
      }

    } else if (n->IsVar()) {
      decltype(op_attrs)* attr;
      if (marked_nodes.count(n)) {
        attr = &marked_var_attrs;
      } else if (const_cast<Node*>(n)->Var() &&
                 const_cast<Node*>(n)->Var()->Persistable()) {
        attr = &param_attrs;
      } else {
        attr = &arg_attrs;
      }

      dot->AddNode(node_id, *attr, node_id);
    }
    node2dot[n] = node_id;
  }
  // Create edges
  for (const Node* n : graph.Nodes()) {
    const auto& src_id = node2dot.at(n);
    for (auto* out : n->outputs) {
      const auto& trg_id = node2dot.at(out);
      dot->AddEdge(src_id, trg_id, {});
    }
  }
}

GraphVizPass::marked_nodes_t GraphVizPass::ConsumeMarkedNodes(
    const Graph* graph) const {
  marked_nodes_t res;
  if (graph->Has(kGraphvizMarkedNodeAttr)) {
    auto& attr = graph->Get<marked_nodes_t>(kGraphvizMarkedNodeAttr);
    res = attr;
    attr.clear();
  }
  return res;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(graph_viz_pass, paddle::framework::ir::GraphVizPass)
    .RequirePassAttr(paddle::framework::ir::kGraphVizPath);
