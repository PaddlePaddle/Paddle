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

#include "paddle/fluid/pir/transforms/general/transfer_layout_pass.h"

#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/passes.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"

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

  Node() : data(pir::Value(nullptr)) {}

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
                            os << "Op(" << op->name() << " " << op << ")";
                          },
                          [&](const pir::Value& value) {
                            if (!value)
                              os << "Var(null)";
                            else
                              os << "Var(" << value.defining_op()->name() << " "
                                 << value.defining_op() << ")";
                          },
                          [&](SrcNode arg) { os << "Src"; },
                          [&](DstNode arg) { os << "Dst"; }},
               n.data);
    return os;
  }
};

SrcNode::operator Node() const { return Node(*this); }

DstNode::operator Node() const { return Node(*this); }

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
    // We assume by default that the program is topologically sorted;
    // otherwise, it will fail during destruction.

    for (auto& op : *(program.block())) {
      Node op_node(&op);
      auto layout_transform_iface =
          op.dyn_cast<paddle::dialect::LayoutTransformationInterface>();
      const auto& relevate_inputs =
          layout_transform_iface ? layout_transform_iface.RelevantInputs(&op)
                                 : op.operands_source();
      const auto& relevate_outputs =
          layout_transform_iface ? layout_transform_iface.RelevantOutputs(&op)
                                 : op.results();
      std::cout << "[BuildGraph]" << op_node
                << " isz:" << relevate_inputs.size()
                << " osz:" << relevate_outputs.size() << std::endl;

      // add in edge
      for (auto& operand : relevate_inputs) {
        Node operand_node(operand);
        // the capacity should be set as the out_degree of operand node
        AddEdge(
            operand_node, op_node, 1.0f / (operand.use_count()), 0.0f, true);
      }

      for (const auto& op_result : relevate_outputs) {
        // we have ssa, so the output must not be processed
        Node op_result_node(op_result);
        AddEdge(op_node, op_result_node, 1.0f, 0.0f, true);
      }
    }

    PreProcess();
  }

  void PreProcess() {
    // the algorithm only accepts two kinds of layout, we assign src node
    // and dst node each a kind. in the begin, each var node have a
    // layout, but the layout of op node is uncertain.

    // TODO(lyk): we need a getLayout interface to get the layout of op /
    // value and determine how many kinds of layout in program currently.
    // Then we call prefer_layout get the total count while running the
    // algorithm. To simplify the experiment, we skip the first step here
    // and just assume they're all NCHW

    for (auto& op : *(program.block())) {
      // we need to ensure the edge from src node to real src node in
      // calculation graph
      //   if (!op.HasTrait<pir::ImmutableLayoutTrait>() &&
      //   op.num_operands() > 0 &&
      //       op.name() != "builtin.combine") {
      if (!op.HasTrait<pir::ImmutableLayoutTrait>() && op.num_operands() > 0) {
        continue;
      }
      Node op_node(&op);
      AddEdge(src_node(), op_node, INF);

      auto layout_transform_iface =
          op.dyn_cast<paddle::dialect::LayoutTransformationInterface>();
      const auto& relevate_inputs =
          layout_transform_iface ? layout_transform_iface.RelevantInputs(&op)
                                 : op.operands_source();
      const auto& relevate_outputs =
          layout_transform_iface ? layout_transform_iface.RelevantOutputs(&op)
                                 : op.results();

      for (const auto& op_operand : relevate_inputs) {
        Node operand_node(op_operand);
        AddEdge(src_node(), operand_node, INF);
      }

      for (const auto& op_result : relevate_outputs) {
        Node op_result_node(op_result);
        AddEdge(src_node(), op_result_node, INF);
      }
    }

    std::unordered_set<Node> nhwc_nodes;
    for (auto& op : *(program.block())) {
      auto layout_transform_iface =
          op.dyn_cast<paddle::dialect::LayoutTransformationInterface>();
      if (!layout_transform_iface) {
        continue;
      }

      auto prefer_layout = layout_transform_iface.PreferLayout(&op);
      if (prefer_layout == common::DataLayout::NHWC) {
        Node op_node(&op);
        nhwc_nodes.insert(op_node);
        AddEdge(op_node, dst_node(), INF);
      }
    }

    // Since VarDesc doesn't store layout, in pir we set all layout to
    // NCHW after translation. However, we need the real layout to decide
    // if we need to alter the operation and value. Here we start from the
    // operation who have a dertermined layout and spread its layout to
    // its output and inputs recursively.
    std::queue<Node> q;
    for (auto& n : nhwc_nodes) {
      q.push(n);
    }
    std::unordered_set<Node> is_node_layout_visited;
    int i = 0;
    while (!q.empty()) {
      std::cout << "before : " << q.size() << " " << i << std::endl;
      i++;
      Node node = q.front();
      std::cout << "visiting node: " << node << std::endl;
      q.pop();
      if (is_node_layout_visited.find(node) != is_node_layout_visited.end()) {
        continue;
      }
      is_node_layout_visited.insert(node);

      std::cout << "judging node: " << node << std::endl;

      auto judge_dense_tensor_type = [](paddle::dialect::DenseTensorType t) {
        if (t.dims().size() == 4) {
          return false;
        }
        return true;
      };

      bool should_interrupt = std::visit(
          overloaded{
              [&](const pir::Operation* op) {
                // TODO(lyk): These conditions may be too loose,
                // we should make a white list here.
                if (op->name() == "pd_op.reshape" ||
                    op->name() == "pd_op.shape" ||
                    op->name() == "pd_op.transpose" ||
                    op->name() == "pd_op.unsqueeze" ||
                    op->name() == "pd_op.squeeze" ||
                    op->name() == "pd_op.flatten") {
                  return true;
                }
                return false;
              },
              [&](const pir::Value& v) {
                if (!v) return true;
                auto vt = v.type();
                if (!vt) return true;
                // maybe not DenseTensor, but we can handle other types
                // later
                if (auto vdt =
                        vt.dyn_cast<paddle::dialect::DenseTensorType>()) {
                  std::cout << "judging var: " << v.defining_op() << " "
                            << v.type() << " " << vdt.dims() << " "
                            << (vdt.dims().size() == 4) << std::endl;
                  return judge_dense_tensor_type(vdt);
                } else if (auto vdt = vt.dyn_cast<pir::VectorType>()) {
                  if (vdt.size() == 0) return false;
                  auto vt_elem = vdt[0];
                  if (auto vdt_elem =
                          vt_elem.dyn_cast<paddle::dialect::DenseTensorType>())
                    return judge_dense_tensor_type(vdt_elem);
                }
                return true;
              },
              [](const auto&) { return true; },
          },
          node.data);
      if (should_interrupt) {
        continue;
      }

      std::cout << "add node to nhwc set: " << node << std::endl;
      nhwc_nodes.insert(node);

      std::cout << "processing node successor: " << node << std::endl;

      int j = 0;
      for (const auto& e : adjs[node]) {
        auto& edge = edges[e];
        q.push(edge.dst);
        std::cout << "add node to queue: " << node << " -> " << edge.dst
                  << std::endl;
        j++;
      }
    }

    q.push(src_node());
    is_node_layout_visited.clear();
    while (!q.empty()) {
      auto node = q.front();
      q.pop();
      if (is_node_layout_visited.find(node) != is_node_layout_visited.end()) {
        continue;
      }
      is_node_layout_visited.insert(node);
      if (nhwc_nodes.count(node) == 0) {
        std::cout << "add node to nchw set: " << node << std::endl;
        AddEdge(src_node(), node, INF);
      }
      for (const auto& e : adjs[node]) {
        auto& edge = edges[e];
        q.push(edge.dst);
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
    std::cout << "--------------------[max flow "
                 "start]---------------------------"
              << std::endl;
    float total_flow = 0.0f;
    while (ConstructLevelGraph()) {
      for (auto& [node, nexts] : adjs) {
        cur_arcs[node] = 0;
      }
      while (auto f = FindBlockingFlow(src_node(), INF)) {
        total_flow += f;
      }
    }
    std::cout << "--------------------[max flow end]---------------------------"
              << std::endl;
    return total_flow;
  }

  std::tuple<std::unordered_set<Node>, std::vector<Edge>> min_cut() {  // NOLINT
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
      std::cout << "bfs access: " << n << std::endl;
      if (src_set.count(n) > 0) continue;
      src_set.insert(n);
      std::cout << "bfs insert " << n << " " << src_set.size() << std::endl;
      for (auto& ind : adjs[n]) {
        std::cout << "bfs edge: " << edges[ind] << " c:" << edges[ind].capacity
                  << " f:" << edges[ind].flow << std::endl;
        if (edges[ind].capacity > edges[ind].flow) {
          std::cout << "bfs add: " << edges[ind].dst << std::endl;
          q.push(edges[ind].dst);
        }
      }
    }

    std::cout << "src_set.size()=" << src_set.size() << std::endl;

    std::vector<Edge> cut;
    for (const auto& e : edges) {
      if (!e.real) continue;
      auto& src = e.src;
      auto& dst = e.dst;
      bool src_cond = (src_set.count(src) > 0);
      bool dst_cond = (src_set.count(dst) > 0);
      if (src_cond == dst_cond) {
        continue;
      }
      std::cout << "cut " << src << "(" << src_cond << ")"
                << " " << dst << "(" << dst_cond << ")" << std::endl;
      cut.push_back(e);
    }

    std::cout << "cut set.size()=" << cut.size() << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    return {src_set, cut};
  }
};

using Edge = FlowGraph::Edge;

class TransferLayoutPass : public pir::Pass {
 public:
  TransferLayoutPass() : pir::Pass("transfer_layout_pass", 3) {}

  void Run(pir::Operation* op) override {
    PADDLE_ENFORCE(
        op->isa<pir::ModuleOp>(),
        common::errors::InvalidArgument(
            "The target of TransferLayoutPass should be a Module Op"));

    pir::IrContext* ctx = pir::IrContext::Instance();
    ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
    ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

    auto module_op = op->dyn_cast<pir::ModuleOp>();
    auto* program = module_op.program();

    std::cout << "---------------------[program before "
                 "pass]---------------------"
              << std::endl;
    std::cout << *program << std::endl;

    // MinCut
    std::cout << "---------------------MinCut---------------------"
              << std::endl;
    FlowGraph graph(*program);
    auto&& [src_set, cut] = graph.min_cut();
    for (auto& e : cut) {
      std::cout << e << std::endl;
    }

    // collect all edges from variable to operation
    // for these, we only need to add 1 for every variable
    // instead of every edges
    std::unordered_map<Node, std::vector<Node>> var_set;
    std::unordered_map<Node, Node> op_src_set;
    std::vector<Edge> op_set;
    for (const auto& e : cut) {
      if (std::get_if<const pir::Operation*>(&(e.src.data))) {
        op_src_set[e.src] = e.dst;
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
        std::cout << op << ",";
      }
      std::cout << std::endl;
    }

    std::cout << "-----------------------[op src set]------------------------"
              << std::endl;

    // cout op_set
    for (auto& [k, v] : op_src_set) {
      std::cout << k << "," << v << std::endl;
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

    std::deque<Node> q;
    std::unordered_set<Node> is_node_layout_visited;
    std::function<void(Node)> topological_visit = [&](Node node) -> void {
      if (is_node_layout_visited.count(node)) return;
      is_node_layout_visited.insert(node);

      // add successors to queue
      for (auto& ind : graph.adjs[node]) {
        auto& e = graph.edges[ind];
        if (!e.real) continue;
        topological_visit(e.dst);
      }

      q.push_front(node);
    };

    for (auto& op : *(program->block())) {
      if (op.num_operands() > 0) continue;
      Node op_node(&op);
      topological_visit(op_node);
      q.push_front(op_node);
    }

    std::cout << "-----------------------[topological "
                 "sort]------------------------"
              << std::endl;

    // process every node by topology and the consequences of min cut
    // while (!q.empty()) {
    //   auto node = q.front();
    //   q.pop_front();
    //   topological_visit(node);
    // }
    // process every node by topology and the consequences of min cut
    // while (!q.empty()) {
    //   auto node = q.front();
    //   q.pop_front();
    //   topological_visit(node);
    // }
    for (auto n : q) {
      std::cout << n << std::endl;
    }

    std::cout << "-----------------------[rewrite "
                 "begin]------------------------"
              << std::endl;

    while (!q.empty()) {
      auto node = q.front();
      q.pop_front();

      for (size_t i = 0; i < 10; i++) std::cout << std::endl;
      if (src_set.find(node) == src_set.end()) {
        std::cout << *program << std::endl;
        if (std::get_if<const pir::Operation*>(&(node.data)) != nullptr) {
          auto* op = const_cast<pir::Operation*>(
              std::get<const pir::Operation*>(node.data));
          for (size_t i = 0; i < 10; i++) std::cout << std::endl;
          std::cout << "[Rewrite][RewriteByLayout] " << node << std::endl;
          std::cout << *program << std::endl;
          auto layout_transformation_iface =
              op->dyn_cast<paddle::dialect::LayoutTransformationInterface>();
          if (layout_transformation_iface) {
            layout_transformation_iface.RewriteByLayout(
                op, common::DataLayout::NHWC);
          } else {
            PADDLE_THROW(common::errors::Unimplemented(
                "Op %s should have a specialized RewriteByLayout "
                "function",
                op->name()));
          }
        }
      }

      std::cout << "[Rewrite] for " << node << std::endl;
      // if node is the src node of a cut edge
      // and it's an operation
      if (op_src_set.find(node) != op_src_set.end()) {
        std::cout << "[Rewrite][Op] for " << node << std::endl;

        // just insert a transpose op
        auto src = node;
        auto dst = op_src_set[src];
        auto dst_value = std::get<pir::Value>(dst.data);

        std::cout << "[Rewrite][Op] for var:"
                  << (dst_value ? (dst_value.defining_op()) : nullptr)
                  << " t:" << (dst_value ? (dst_value.type()) : pir::Type())
                  << std::endl;
        transpose_op->set_attribute(
            "source",
            pir::StrAttribute::get(transpose_op->ir_context(),
                                   "transfer_layout_pass"));
        // TODO(lyk): special process for reshape, we cannot only just
        // insert a transpose op temporarily ignore reshape if
        // (dst_value.defining_op()->name() == "pd_op.reshape") continue;

        // enforce dst value.defining_op = src
        ((src_set.count(node) > 0) ? layout_to_perm("NCHW", "NHWC")
                                   : layout_to_perm("NHWC", "NCHW"));
        builder.SetInsertionPointAfter(dst_value.defining_op());
        auto transpose_op =
            builder.Build<paddle::dialect::TransposeOp>(dst_value, perm);
        auto replace_uses_without_self = [&](pir::OpOperand arg) {
          return arg.owner() != transpose_op.operation();
        };
        dst_value.ReplaceUsesWithIf(transpose_op.out(),
                                    replace_uses_without_self);
      }

      // if node is the src node of a cut edge
      // and it's a value
      // this node must not be in the nhwc set
      if (var_set.find(node) != var_set.end()) {
        std::cout << "[Rewrite][Var] for " << node << std::endl;
        const auto& ops = var_set[node];
        // operand should be replaced
        std::unordered_set<const pir::Operation*> operation_set;
        for (auto op : ops) {
          operation_set.insert(std::get<const pir::Operation*>(op.data));
        }

        auto value = std::get<pir::Value>(node.data);
        std::cout << "[Rewrite][Var] for var:"
                  << (value ? value.defining_op() : nullptr);
        for (const auto& op : operation_set) {
          std::cout << " op: " << op << ",";
        }
        std::cout << std::endl;
        const auto& perm =
                                       : layout_to_perm("NHWC", "NCHW"));
        builder.SetInsertionPointAfter(value.defining_op());
        auto transpose_op =
            builder.Build<paddle::dialect::TransposeOp>(value, perm);
        auto replace_uses_in_cut_set = [&](pir::OpOperand arg) {
          return (operation_set.find(arg.owner()) != operation_set.end()) &&
                 (arg.owner() != transpose_op.operation());
        };
        value.ReplaceUsesWithIf(transpose_op.out(), replace_uses_in_cut_set);
      }
    }

    std::cout << "---------------------[program after "
                 "pass]---------------------"
              << std::endl;
    std::cout << *program << std::endl;
  }
};

namespace pir {

std::unique_ptr<pir::Pass> CreateTransferLayoutPass() {
  return std::make_unique<TransferLayoutPass>();
}

}  // namespace pir

REGISTER_IR_PASS(transfer_layout_pass, TransferLayoutPass);
