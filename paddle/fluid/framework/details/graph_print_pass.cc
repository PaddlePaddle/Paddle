// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/graph_print_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

class GraphvizVar : public GraphvizNode {
 public:
  GraphvizVar(ir::Node* n, const int& i) : GraphvizNode(n, i) {}
  friend std::ostream& operator<<(std::ostream& sout, const GraphvizVar& var) {
    sout << "var_" << var.id_ << " [label=\"" << var.node_->Name() << "\"]"
         << std::endl;
    return sout;
  }
};

class GraphvizOp : public GraphvizNode {
 public:
  GraphvizOp(ir::Node* n, const int& i) : GraphvizNode(n, i) {}
  friend std::ostream& operator<<(std::ostream& sout, const GraphvizOp& op) {
    sout << "op_" + std::to_string(op.id_) << " [label=\"" << op.node_->Name()
         << "\", shape=rect]" << std::endl;
    PADDLE_ENFORCE(op.stream_.rdbuf()->in_avail() != 0,
                   "No inputs outputs. Please call AddEdge first!");
    sout << op.stream_.str();
    return sout;
  }
  template <typename Callback>
  void AddEdge(const Callback& cb) {
    std::string op_name = "op_" + std::to_string(id_);
    for (auto var : node_->inputs) {
      std::string var_name = "var_" + std::to_string(cb(var));
      stream_ << var_name << "->" << op_name << std::endl;
    }
    for (auto var : node_->outputs) {
      std::string var_name = "var_" + std::to_string(cb(var));
      stream_ << op_name << "->" << var_name << std::endl;
    }
  }

  template <typename Callback>
  void AddCustomEdge(const Callback& cb) {
    stream_ << cb() << std::endl;
  }

 private:
  std::ostringstream stream_;
};

template <typename T, typename Container>
std::vector<T*> FilterByNodeWrapper(const Container& con) {
  std::vector<T*> ret;
  for (auto& node : con) {
    auto i = dynamic_cast<T*>(node.get());
    if (i != nullptr) ret.emplace_back(i);
  }
  return ret;
}

std::unordered_map<ir::Node*, int> SSAGraphPrinterImpl::ToGraphvizNode(
    const ir::Graph& graph) const {
  // Convert to GraphvizNode format
  auto& graphviz_nodes = graph.Get<GraphvizNodes>(kGraphviz);
  graphviz_nodes.clear();
  std::unordered_map<ir::Node*, int> vars;
  std::unordered_map<ir::Node*, GraphvizOp*> ops;
  int var_id = 0;
  int op_id = 0;
  for (auto& node : graph.Nodes()) {
    if (node->IsVar()) {
      graphviz_nodes.emplace(new GraphvizVar(node, var_id));
      vars.emplace(std::make_pair(node, var_id++));
    } else if (node->IsOp()) {
      std::unique_ptr<GraphvizOp> op(new GraphvizOp(node, op_id++));
      ops[node] = op.get();
      graphviz_nodes.emplace(std::move(op));
    } else {
      PADDLE_THROW("Unknown op type");
    }
  }

  // Detect circle. Draw circle in different lines
  std::vector<std::vector<ir::Node*>> circles;
  const std::string kCircleEdge = "[color=red,penwidth=3.0]";
  if (ir::FindCircleSubGraph(graph, &circles)) {
    VLOG(3) << "Graph has circle! circles count : " << circles.size();
    for (auto& circle : circles) {
      for (size_t i = 0; i < circle.size() - 1; ++i) {
        GraphvizOp* prev = ops[circle[i]];
        GraphvizOp* next = ops[circle[i + 1]];
        std::string prev_op = "op_" + std::to_string(prev->Id());
        std::string next_op = "op_" + std::to_string(next->Id());
        prev->AddCustomEdge([&]() -> std::string {
          return prev_op + "->" + next_op + kCircleEdge;
        });
      }
    }
  }
  return vars;
}

void SSAGraphPrinterImpl::Print(const ir::Graph& graph,
                                std::ostream& sout) const {
  auto vars = ToGraphvizNode(graph);
  auto& nodes = graph.Get<GraphvizNodes>(kGraphviz);

  sout << "digraph G {\n";
  for (auto& var : FilterByNodeWrapper<GraphvizVar>(nodes)) {
    sout << *var;
  }

  for (auto& op : FilterByNodeWrapper<GraphvizOp>(nodes)) {
    op->AddEdge([&vars](ir::Node* var) { return vars.at(var); });
    sout << *op;
  }
  sout << "}\n";
}

std::unique_ptr<ir::Graph> SSAGraphPrintPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  printer_.reset(new SSAGraphPrinterImpl());
  std::unique_ptr<std::ostream> fout(
      new std::ofstream(Get<std::string>(kGraphvizPath)));
  PADDLE_ENFORCE(fout->good() == true, "Failed to open file.");

  printer_->Print(*graph, *fout);
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(graph_print_pass, paddle::framework::details::SSAGraphPrintPass)
    .RequirePassAttr(paddle::framework::details::kGraphvizPath);
