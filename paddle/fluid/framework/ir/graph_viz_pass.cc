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

#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
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

ir::Graph* GraphVizPass::ApplyImpl(ir::Graph* graph) const {
  const std::string graph_viz_path = Get<std::string>(kGraphVizPath);
  VLOG(3) << "draw IR graph viz to " << graph_viz_path;
  std::unique_ptr<std::ostream> fout(new std::ofstream(graph_viz_path));
  PADDLE_ENFORCE(fout->good());
  std::ostream& sout = *fout;

  std::unordered_map<const ir::Node*, std::string> node2dot;

  Dot dot;

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

  auto marked_nodes = ConsumeMarkedNodes(graph);
  // Create nodes
  for (const Node* n : graph->Nodes()) {
    std::string node_id = FormatName(n) + "(" + std::to_string(n->id()) + ")";
    if (n->IsOp()) {
      decltype(op_attrs) attr =
          marked_nodes.count(n) ? marked_op_attrs : op_attrs;
      dot.AddNode(node_id, attr, node_id);
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

      dot.AddNode(node_id, *attr, node_id);
    }
    node2dot[n] = node_id;
  }
  // Create edges
  for (const Node* n : graph->Nodes()) {
    const auto& src_id = node2dot.at(n);
    for (auto* out : n->outputs) {
      const auto& trg_id = node2dot.at(out);
      dot.AddEdge(src_id, trg_id, {});
    }
  }

  sout << dot.Build();

  return graph;
}

GraphVizPass::marked_nodes_t GraphVizPass::ConsumeMarkedNodes(
    Graph* graph) const {
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
