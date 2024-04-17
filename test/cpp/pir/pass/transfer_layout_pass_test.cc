// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <gtest/internal/gtest-port.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "paddle/common/layout.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/passes.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass_manager.h"

namespace {

class Node;

struct SrcNode {
  bool operator==(const SrcNode& rhs) const { return true; }
  operator Node() const;
  friend std::ostream& operator<<(std::ostream& os, const SrcNode& n) {
    os << "Src";
    return os;
  }
};
struct DstNode {
  bool operator==(const DstNode& rhs) const { return true; }
  operator Node() const;
  friend std::ostream& operator<<(std::ostream& os, const DstNode& n) {
    os << "Dst";
    return os;
  }
};

SrcNode src_node() { return SrcNode(); }
DstNode dst_node() { return DstNode(); }

const float INF = std::numeric_limits<float>::max();
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

struct Node {
  using DataType =
      std::variant<const pir::Operation*, pir::Value, SrcNode, DstNode>;
  DataType data;

  explicit Node(const pir::Operation* op) : data(op) {}
  explicit Node(pir::Value value) : data(value) {}
  explicit Node(SrcNode n) : data(n) {}
  explicit Node(DstNode n) : data(n) {}

  bool operator==(const Node& rhs) const {
    bool ret = std::visit(
        overloaded{
            [](const pir::Operation* left, const pir::Operation* right) {
              return (left == right);
            },
            [](const pir::Value& left, const pir::Value& right) {
              return (left == right);
            },
            [](const SrcNode& left, const SrcNode& right) { return true; },
            [](const DstNode& left, const DstNode& right) { return true; },
            [](auto& left, auto& right) { return false; }},
        data,
        rhs.data);
    return ret;
  }
  friend std::ostream& operator<<(std::ostream& os, const Node& n) {
    std::visit(overloaded{[&](const pir::Operation* op) {
                            os << "Op(" << op->name() << ")";
                          },
                          [&](const pir::Value& value) {
                            os << "Var(" << value.defining_op()->name() << ")";
                          },
                          [&](SrcNode arg) { os << "Src"; },
                          [&](DstNode arg) { os << "Dst"; }},
               n.data);
    return os;
  }
};

SrcNode::operator Node() const { return Node(*this); }

DstNode::operator Node() const { return Node(*this); }

}  // namespace

namespace std {

template <>
struct hash<SrcNode> {
  size_t operator()(const Node& s) const noexcept { return 0x111; }
};

template <>
struct hash<DstNode> {
  size_t operator()(const Node& s) const noexcept { return 0x222; }
};

template <>
struct hash<Node> {
  size_t operator()(const Node& s) const noexcept {
    return hash<Node::DataType>{}(s.data);
  }
};

}  // namespace std

namespace {
struct FlowGraph {
  using EdgeIndex = size_t;

  struct Edge {
    Node src;
    Node dst;
    float capacity;
    float flow;
    bool real;

    Edge(Node src,
         Node dst,
         float capacity = 0.0f,
         float flow = 0.0f,
         bool real = false)
        : src(src), dst(dst), capacity(capacity), flow(flow), real(real) {}
    friend std::ostream& operator<<(std::ostream& os, const Edge& n) {
      os << "(" << n.src << "->" << n.dst << ")";
      return os;
    }
  };

  std::vector<Edge> edges;
  // std::vector<Node> nodes;
  std::unordered_map<Node, std::vector<EdgeIndex>> adjs;
  std::unordered_map<Node, EdgeIndex> cur_arcs;
  std::unordered_map<Node, size_t> heights;
  const pir::Program& program;

  void AddEdge(Node src,
               Node dst,
               float capacity = 0.0f,
               float flow = 0.0f,
               bool real = false) {
    if (src == dst) {
      return;
    }

    edges.emplace_back(src, dst, capacity, flow, real);
    adjs[src].push_back(edges.size() - 1);

    // add reverse edge
    edges.emplace_back(dst, src, 0, flow);
    adjs[dst].push_back(edges.size() - 1);
  }

  explicit FlowGraph(const pir::Program& program) : program(program) {
    // We assume by default that the program is topologically sorted; otherwise,
    // it will fail during destruction.

    for (const auto& op : *(program.block())) {
      Node op_node(&op);
      // add in edge
      for (auto& operand : op.operands_source()) {
        Node operand_node(operand);
        // the capacity should be set as the out_degree of operand node
        AddEdge(
            operand_node, op_node, 1.0f / (operand.use_count()), 0.0f, true);
      }

      for (const auto& op_result : op.results()) {
        // we have ssa, so the output must not be processed
        Node op_result_node(op_result);
        AddEdge(op_node, op_result_node, 1.0f, 0.0f, true);
      }
    }

    PreProcess();
  }

  void PreProcess() {
    // the algorithm only accepts two kinds of layout, we assign src node and
    // dst node each a kind. in the begin, each var node have a layout, but the
    // layout of op node is uncertain.

    // TODO(lyk): we need a getLayout interface to get the layout of op /
    // value and determine how many kinds of layout in program currently. Then
    // we call prefer_layout get the total count while running the algorithm. To
    // simplify the experiment, we skip the first step here and just assume
    // they're all NCHW

    for (const auto& op : *(program.block())) {
      if (!op.HasTrait<pir::ImmutableLayoutTrait>()) {
        continue;
      }
      Node op_node(&op);
      AddEdge(src_node(), op_node, INF);

      for (const auto& op_operand : op.operands_source()) {
        Node operand_node(op_operand);
        AddEdge(src_node(), operand_node, INF);
      }

      for (const auto& op_result : op.results()) {
        Node op_result_node(op_result);
        AddEdge(src_node(), op_result_node, INF);
      }
    }

    for (auto& op : *(program.block())) {
      auto layout_transform_iface =
          op.dyn_cast<paddle::dialect::LayoutTransformationInterface>();
      if (!layout_transform_iface) {
        continue;
      }

      auto prefer_layout = layout_transform_iface.PreferLayout(&op);
      if (prefer_layout == common::DataLayout::NHWC) {
        Node op_node(&op);
        AddEdge(op_node, dst_node(), INF);
      }
    }
  }

  bool ConstructLevelGraph() {
    heights.clear();
    std::queue<std::pair<Node, size_t>> q;
    q.push({src_node(), 0});
    while (!q.empty()) {
      auto [node, height] = q.front();
      q.pop();
      if (heights.find(node) != heights.end()) {
        continue;
      }
      heights[node] = height;
      for (auto e_ind : adjs[node]) {
        auto& e = edges[e_ind];
        if (e.capacity - e.flow > 0 && heights.find(e.dst) == heights.end()) {
          q.push({e.dst, height + 1});
        }
      }
    }
    return (heights[dst_node()] > 0);
  }

  // cf is the admissable flow in current path
  float FindBlockingFlow(Node src, float cf) {
    if (src == dst_node() || abs(cf) < 1e-9) {
      return cf;
    }

    auto& next_arc = cur_arcs[src];  // notice this is a reference
    float ret = 0.0f;
    while (next_arc < adjs[src].size()) {
      auto e_ind = adjs[src][next_arc];
      auto& e = edges[e_ind];
      next_arc++;
      auto next_node = e.dst;
      if (heights[next_node] == heights[src] + 1) {
        auto left_capacity = e.capacity - e.flow;
        auto update_flow = std::min(cf - ret, left_capacity);
        auto f = FindBlockingFlow(next_node, update_flow);
        if (f > 0) {
          e.flow += f;
          auto reverse_e_ind = e_ind ^ 1;
          auto& reverse_e = edges[reverse_e_ind];
          reverse_e.flow -= f;
          ret += f;

          if (abs(ret - cf) < 1e-9) {
            return ret;
          }
        }
      }
    }

    if (ret == 0) {
      heights[src] = 0;
    }

    return ret;
  }

  float max_flow() {
    float total_flow = 0.0f;
    while (ConstructLevelGraph()) {
      for (auto& [node, nexts] : adjs) {
        cur_arcs[node] = 0;
      }
      while (auto f = FindBlockingFlow(src_node(), INF)) {
        total_flow += f;
      }
    }
    return total_flow;
  }

  std::vector<Edge> min_cut() {  // NOLINT
    max_flow();
    // from src_node get its reachable nodes and call them S
    // other nodes are in T
    // collect edges between S and T
    std::unordered_set<Node> src_set;
    std::queue<Node> q;
    q.push(src_node());
    while (!q.empty()) {
      auto n = q.front();
      q.pop();
      if (src_set.count(n) > 0) continue;
      src_set.insert(n);
      std::cout << "bfs accesss" << n << " " << src_set.size() << std::endl;
      for (auto& ind : adjs[n]) {
        if (edges[ind].capacity - edges[ind].flow > 0) {
          q.push(edges[ind].dst);
        }
      }
      q.pop();
    }

    std::cout << "src_set.size()=" << src_set.size() << std::endl;

    std::vector<Edge> cut;
    for (auto& n : src_set) {
      std::cout << "adj[" << n << "].size()=" << adjs[n].size() << std::endl;
      for (auto& ind : adjs[n]) {
        auto& e = edges[ind];
        if (!e.real) continue;
        cut.push_back(e);
      }
    }

    std::cout << "cut set.size()=" << cut.size() << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    return cut;
  }
};

using Edge = FlowGraph::Edge;

}  // namespace

using ProgramDesc = paddle::framework::ProgramDesc;
ProgramDesc load_from_file(const std::string& file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);

  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());  // NOLINT
  fin.close();
  return ProgramDesc(buffer);
}

TEST(transfer_layout_pass, pass_test) {
  // Load Unet Program
  const std::string model_name = "sd15_unet.pdmodel";
  auto p = load_from_file(model_name);
  EXPECT_EQ(p.Size(), 1u);
  EXPECT_GT(p.Block(0).OpSize(), 0u);

  // Translate to PIR Program
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);
  std::cout << *program << std::endl;

  pir::PassManager pass_pm(::pir::IrContext::Instance(), 3);
  for (const auto& gpu_pass : paddle::kPirGpuPasses) {
    pass_pm.AddPass(pir::PassRegistry::Instance().Get(gpu_pass));
  }
  pass_pm.Run(program.get());
  std::cout << *program << std::endl;

  // MinCut
  std::cout << "---------------------MinCut---------------------" << std::endl;
  FlowGraph graph(*program);
  auto cut = graph.min_cut();
  for (auto& e : cut) {
    std::cout << e << std::endl;
  }

  // collect all edges from variable to operation
  // for these, we only need to add 1 for every variable
  // instead of every edges
  std::unordered_map<Node, std::vector<Node>> var_set;
  std::vector<Edge> op_set;
  for (const auto& e : cut) {
    if (std::get_if<const pir::Operation*>(&(e.src.data))) {
      op_set.push_back(e);
    } else {
      var_set[e.src].push_back(e.dst);
    }
  }

  std::cout << "-----------------------[var set]------------------------"
            << std::endl;

  // cout var_set
  for (auto& [var, ops] : var_set) {
    std::cout << var << ":";
    for (auto op : ops) {
      std::cout << "op," << op;
    }
    std::cout << std::endl;
  }

  std::cout << "-----------------------[op set]------------------------"
            << std::endl;

  // cout op_set
  for (auto& e : op_set) {
    std::cout << e << std::endl;
  }

  std::cout << "-----------------------[min cut end]------------------------"
            << std::endl;

  pir::Builder builder(ctx, program->block());
  auto layout_to_perm = [](std::string src, std::string dst) {
    std::vector<int> perm(src.size(), 0);
    std::unordered_map<char, int> d;
    for (size_t i = 0; i < src.size(); ++i) {
      d[src[i]] = i;
    }
    for (size_t i = 0; i < dst.size(); ++i) {
      perm[i] = d[dst[i]];
    }
    return perm;
  };
  for (auto& [var, ops] : var_set) {
    std::unordered_set<const pir::Operation*> operation_set;
    for (auto op : ops) {
      operation_set.insert(std::get<const pir::Operation*>(op.data));
    }
    auto replace_uses_in_cut_set = [&](pir::OpOperand arg) {
      return operation_set.find(arg.owner()) != operation_set.end();
    };
    auto value = std::get<pir::Value>(var.data);
    auto transpose_op = builder.Build<paddle::dialect::TransposeOp>(
        value, layout_to_perm("NCHW", "NHWC"));
    value.ReplaceUsesWithIf(transpose_op.out(), replace_uses_in_cut_set);
  }

  std::cout << "-----------------------[trans var end]------------------------"
            << std::endl;

  // insert transpose between
}
