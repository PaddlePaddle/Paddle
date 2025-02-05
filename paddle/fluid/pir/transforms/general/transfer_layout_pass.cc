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
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pass/utils.h"

struct Node;

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

// for some edge weight, we need to set it to INF, but which
// may cause precision problem. So we choose a value
// large enough.
const float THRESHOLD = INF / 2.0f;

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
  std::unordered_map<Node, std::vector<EdgeIndex>> adjs;

  // used to avoid duplicate links
  std::unordered_map<Node, std::unordered_set<Node>> adjs_by_node;
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

    if (adjs_by_node[src].find(dst) != adjs_by_node[src].end()) {
      return;
    }

    edges.emplace_back(src, dst, capacity, flow, real);
    adjs[src].push_back(edges.size() - 1);
    adjs_by_node[src].insert(dst);

    // add reverse edge
    edges.emplace_back(dst, src, 0, flow);
    adjs[dst].push_back(edges.size() - 1);
    adjs_by_node[dst].insert(src);
  }

  explicit FlowGraph(const pir::Program& program) : program(program) {
    // We assume by default that the program is topologically sorted;
    // otherwise, it will fail during destruction.

    for (auto& op : *(program.block())) {
      Node op_node(&op);
      auto layout_transform_iface =
          op.dyn_cast<paddle::dialect::LayoutTransformationInterface>();
      const auto& relevant_inputs =
          layout_transform_iface ? layout_transform_iface.RelevantInputs(&op)
                                 : op.operands_source();
      const auto& relevant_outputs =
          layout_transform_iface ? layout_transform_iface.RelevantOutputs(&op)
                                 : op.results();
      VLOG(10) << "[BuildGraph]" << op_node << " isz:" << relevant_inputs.size()
               << " osz:" << relevant_outputs.size();

      // add in edge
      for (auto& operand : relevant_inputs) {
        Node operand_node(operand);
        // the capacity should be set as the out_degree of operand node
        float weight = 1.0f;
        if (operand && operand.type()) {
          weight = 1.0f / (operand.use_count());
          if (auto t = operand.type().dyn_cast<pir::VectorType>()) {
            weight = INF;
          }
        }
        AddEdge(operand_node, op_node, weight, 0.0f, true);
      }

      for (const auto& op_result : relevant_outputs) {
        // we have ssa, so the output must not be processed
        Node op_result_node(op_result);

        float weight = 1.0f;
        if (op_result && op_result.type()) {
          if (auto t = op_result.type().dyn_cast<pir::VectorType>()) {
            weight = INF;
          }
        }
        AddEdge(op_node, op_result_node, weight, 0.0f, true);
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

      if (!op.HasTrait<pir::ImmutableLayoutTrait>() && op.num_operands() > 0) {
        continue;
      }
      Node op_node(&op);
      AddEdge(src_node(), op_node, THRESHOLD);

      auto layout_transform_iface =
          op.dyn_cast<paddle::dialect::LayoutTransformationInterface>();
      const auto& relevant_inputs =
          layout_transform_iface ? layout_transform_iface.RelevantInputs(&op)
                                 : op.operands_source();
      const auto& relevant_outputs =
          layout_transform_iface ? layout_transform_iface.RelevantOutputs(&op)
                                 : op.results();

      for (const auto& op_operand : relevant_inputs) {
        Node operand_node(op_operand);
        AddEdge(src_node(), operand_node, THRESHOLD);
      }

      for (const auto& op_result : relevant_outputs) {
        Node op_result_node(op_result);
        AddEdge(src_node(), op_result_node, THRESHOLD);
      }
    }

    std::unordered_set<Node> mutable_nodes;
    for (auto& op : *(program.block())) {
      auto layout_transform_iface =
          op.dyn_cast<paddle::dialect::LayoutTransformationInterface>();
      if (!layout_transform_iface) {
        continue;
      }

      if (!layout_transform_iface.CanBeModified(&op)) {
        continue;
      }

      auto prefer_layout = layout_transform_iface.PreferLayout(&op);
      if (prefer_layout == common::DataLayout::NHWC) {
        Node op_node(&op);
        mutable_nodes.insert(op_node);
        AddEdge(op_node, dst_node(), THRESHOLD);
        VLOG(10) << "[PreProcess] node: " << op_node
                 << " should be set to NHWC";
      }
    }

    // (%397) = "pd_op.isinf" (%398)
    // (%401) = "pd_op.if" (%400)
    //     (%402) = "pd_op.clip" (%397, %3, %2) {}
    //     () = "cf.yield" (%402) {}
    // } else {
    //     () = "cf.yield" (%1) {}
    // }
    // just like the example above, we cannot build full
    // dependency while graph containing control flow
    // so we need a special process
    for (auto& op : *(program.block())) {
      auto layout_transform_iface =
          op.dyn_cast<paddle::dialect::LayoutTransformationInterface>();
      const auto& relevant_outputs =
          layout_transform_iface ? layout_transform_iface.RelevantOutputs(&op)
                                 : op.results();

      for (const auto& op_result : relevant_outputs) {
        Node op_result_node(op_result);
        for (auto it = op_result.use_begin(); it != op_result.use_end(); ++it) {
          auto user_op = it->owner();
          while (user_op->GetParentProgram()->block() != user_op->GetParent()) {
            user_op = user_op->GetParentOp();
          }
          // if user_op is in main block, no additional process
          if (user_op == it->owner()) {
            continue;
          }
          // else user_op in main_block, link it with its producer
          Node user_op_node(user_op);
          VLOG(10) << "[PreProcess] control flow link:" << op_result_node
                   << " -> " << user_op_node;
          AddEdge(op_result_node, user_op_node, 1.0f, 0.0f, true);
        }
      }
    }

    // Since VarDesc doesn't store layout, in pir we set all layout to
    // NCHW after translation. However, we need the real layout to decide
    // if we need to alter the operation and value. Here we start from the
    // operation who have a dertermined layout and spread its layout to
    // its output and inputs recursively.
    std::queue<Node> q;
    for (auto& n : mutable_nodes) {
      q.push(n);
    }
    std::unordered_set<Node> is_node_layout_visited;
    int i = 0;
    while (!q.empty()) {
      VLOG(10) << "before : " << q.size() << " " << i;
      i++;
      Node node = q.front();
      VLOG(10) << "visiting node: " << node;
      q.pop();
      if (is_node_layout_visited.find(node) != is_node_layout_visited.end()) {
        continue;
      }
      is_node_layout_visited.insert(node);

      VLOG(10) << "judging node: " << node;

      auto judge_dense_tensor_type = [](paddle::dialect::DenseTensorType t) {
        if (t.dims().size() == 4) {
          return true;
        }
        return false;
      };

      bool should_interrupt = std::visit(
          overloaded{
              [&](const pir::Operation* op) {
                pir::Operation* fop = const_cast<pir::Operation*>(op);

                auto layout_transform_iface = fop->dyn_cast<
                    paddle::dialect::LayoutTransformationInterface>();
                if (layout_transform_iface) {
                  return !layout_transform_iface.CanBeModified(fop);
                }
                return true;
              },
              [&](const pir::Value& v) {
                if (!v) return true;
                auto vt = v.type();
                if (!vt) return true;
                // maybe not DenseTensor, but we can handle other types later
                bool can_be_transformed = false;
                if (auto vdt =
                        vt.dyn_cast<paddle::dialect::DenseTensorType>()) {
                  VLOG(10) << "judging var: " << v.defining_op() << " "
                           << v.type() << " " << vdt.dims();
                  can_be_transformed = judge_dense_tensor_type(vdt);
                } else if (auto vdt = vt.dyn_cast<pir::VectorType>()) {
                  if (vdt.size() == 0) return false;
                  auto vt_elem = vdt[0];
                  if (auto vdt_elem =
                          vt_elem.dyn_cast<paddle::dialect::DenseTensorType>())
                    can_be_transformed = judge_dense_tensor_type(vdt_elem);
                }
                if (!can_be_transformed) {
                  // when the rank of value is not 4, we can't allow it to be
                  // a point of cut edge. So we set its outputs and inputs to
                  // immutable.
                  Node in_node = Node(v.defining_op());
                  mutable_nodes.erase(in_node);
                  VLOG(10) << "erase node: " << in_node << " from mutable set";

                  for (auto it = v.use_begin(); it != v.use_end(); ++it) {
                    Node out_node(it->owner());
                    mutable_nodes.erase(out_node);
                    VLOG(10)
                        << "erase node: " << out_node << " from mutable set";
                  }
                }
                return !can_be_transformed;
              },
              [](const auto&) { return true; },
          },
          node.data);
      if (should_interrupt) {
        continue;
      }

      VLOG(10) << "add node to mutable set: " << node;
      mutable_nodes.insert(node);

      VLOG(10) << "processing node successor: " << node;

      int j = 0;
      for (const auto& e : adjs[node]) {
        auto& edge = edges[e];
        q.push(edge.dst);
        VLOG(10) << "add node to queue: " << node << " -> " << edge.dst;
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
      if (mutable_nodes.count(node) == 0) {
        VLOG(10) << "add node to nchw set: " << node;
        AddEdge(src_node(), node, THRESHOLD);
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

  // cf is the admissible flow in current path
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
        VLOG(10) << "find blocking flow: " << src << " -> " << next_node
                 << " flow: " << f << " capa: " << left_capacity
                 << " update_flow: " << update_flow;
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

  float MaxFlow() {
    VLOG(10)
        << "--------------------[max flow start]---------------------------";
    float total_flow = 0.0f;
    while (ConstructLevelGraph()) {
      for (auto& [node, nexts] : adjs) {
        cur_arcs[node] = 0;
      }
      VLOG(10) << "--------------------[find blocking flow "
                  "start]---------------------------";
      while (auto f = FindBlockingFlow(src_node(), INF)) {
        total_flow += f;
        VLOG(10) << "[new flow]: " << f;
      }
      VLOG(10) << "--------------------[find blocking flow end]" << total_flow
               << "---------------------------";
    }
    VLOG(10) << "--------------------[max flow end]---------------------------";
    return total_flow;
  }

  std::tuple<std::unordered_set<Node>, std::vector<Edge>> MinCut() {  // NOLINT
    MaxFlow();
    // from src_node get its reachable nodes and call them S
    // other nodes are in T
    // collect edges between S and T
    std::unordered_set<Node> src_set;
    std::queue<Node> q;
    q.push(src_node());
    while (!q.empty()) {
      auto n = q.front();
      q.pop();
      VLOG(10) << "bfs access: " << n;
      if (src_set.count(n) > 0) continue;
      src_set.insert(n);
      VLOG(10) << "bfs insert " << n << " " << src_set.size();
      for (auto& ind : adjs[n]) {
        VLOG(10) << "bfs edge: " << edges[ind] << " c:" << edges[ind].capacity
                 << " f:" << edges[ind].flow;
        if (edges[ind].capacity > edges[ind].flow) {
          VLOG(10) << "bfs add: " << edges[ind].dst;
          q.push(edges[ind].dst);
        }
      }
    }

    VLOG(10) << "src_set.size()=" << src_set.size();

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
      VLOG(10) << "cut " << src << "(" << src_cond << ")"
               << " " << dst << "(" << dst_cond << ")";
      cut.push_back(e);
    }

    VLOG(10) << "cut set.size()=" << cut.size();
    VLOG(10) << "-----------------------------------------------";

    return {src_set, cut};
  }
};

using Edge = FlowGraph::Edge;

class TransferLayoutPass : public pir::Pass {
 public:
  TransferLayoutPass() : pir::Pass("transfer_layout_pass", 2) {}

  bool CanApplyOn(pir::Operation* op) const override {
    if (!op->isa<pir::ModuleOp>()) {
      return false;
    }
    return op->num_regions() > 0;
  }

  void Run(pir::Operation* op) override {
    pir::IrContext* ctx = pir::IrContext::Instance();
    ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
    ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

    auto module_op = op->dyn_cast<pir::ModuleOp>();
    auto* program = module_op.program();

    // MinCut
    VLOG(10) << "---------------------MinCut---------------------";
    FlowGraph graph(*program);
    auto&& [src_set, cut] = graph.MinCut();
    for (auto& e : cut) {
      VLOG(10) << e;
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

    VLOG(10) << "-----------------------[var set]------------------------";

    // cout var_set
    for (auto& [var, ops] : var_set) {
      VLOG(10) << var << ":";
      for (auto op : ops) {
        VLOG(10) << op << ",";
      }
      VLOG(10);
    }

    VLOG(10) << "-----------------------[op src set]------------------------";

    // cout op_set
    for (auto& [k, v] : op_src_set) {
      VLOG(10) << k << "," << v;
    }

    VLOG(10) << "-----------------------[min cut end]------------------------";

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

    VLOG(10)
        << "-----------------------[topological sort]------------------------";

    for (auto n : q) {
      VLOG(10) << n;
    }

    VLOG(10)
        << "-----------------------[rewrite begin]------------------------";
    int64_t num_of_layout_changed_ops{0};
    int64_t num_of_transpose_ops{0};
    while (!q.empty()) {
      auto node = q.front();
      q.pop_front();

      // not in cut set and its layout should not be changed
      if (src_set.find(node) == src_set.end()) {
        // process layout transformation
        if (std::get_if<const pir::Operation*>(&(node.data)) != nullptr) {
          auto* op = const_cast<pir::Operation*>(
              std::get<const pir::Operation*>(node.data));
          VLOG(10) << "[Rewrite][RewriteByLayout] " << node;
          auto layout_transformation_iface =
              op->dyn_cast<paddle::dialect::LayoutTransformationInterface>();
          if (layout_transformation_iface) {
            layout_transformation_iface.RewriteByLayout(
                op, common::DataLayout::NHWC);
            num_of_layout_changed_ops++;
          } else {
            PADDLE_THROW(common::errors::Unimplemented(
                "Op %s should have a specialized RewriteByLayout function",
                op->name()));
          }
        }
      }

      VLOG(10) << "[Rewrite] for " << node;
      // if node is the src node of a cut edge
      // and it's an operation
      if (op_src_set.find(node) != op_src_set.end()) {
        VLOG(10) << "[Rewrite][Op] for " << node;

        // just insert a transpose op
        auto src = node;
        auto dst = op_src_set[src];
        auto dst_value = std::get<pir::Value>(dst.data);

        VLOG(10) << "[Rewrite][Op] for var:"
                 << (dst_value ? (dst_value.defining_op()) : nullptr)
                 << " t:" << (dst_value ? (dst_value.type()) : pir::Type());

        // enforce dst value.defining_op = src
        const auto& perm =
            ((src_set.count(node) > 0) ? layout_to_perm("NCHW", "NHWC")
                                       : layout_to_perm("NHWC", "NCHW"));
        const auto& new_layout =
            ((src_set.count(node) > 0) ? common::DataLayout::NHWC
                                       : common::DataLayout::NCHW);
        builder.SetInsertionPointAfter(dst_value.defining_op());
        num_of_transpose_ops++;
        auto transpose_op =
            builder.Build<paddle::dialect::TransposeOp>(dst_value, perm);
        transpose_op->set_attribute(
            "source",
            pir::StrAttribute::get(transpose_op->ir_context(),
                                   "transfer_layout_pass"));
        auto replace_uses_without_self = [&](pir::OpOperand arg) {
          return arg.owner() != transpose_op.operation();
        };
        pir::SetNewLayoutForValue(transpose_op.out(), new_layout);
        dst_value.ReplaceUsesWithIf(transpose_op.out(),
                                    replace_uses_without_self);
      }

      // if node is the src node of a cut edge
      // and it's a value
      // this node must not be in the nhwc set
      if (var_set.find(node) != var_set.end()) {
        VLOG(10) << "[Rewrite][Var] for " << node;
        const auto& ops = var_set[node];
        // operand should be replaced
        std::unordered_set<const pir::Operation*> operation_set;
        for (auto op : ops) {
          operation_set.insert(std::get<const pir::Operation*>(op.data));
        }

        auto value = std::get<pir::Value>(node.data);
        VLOG(10) << "[Rewrite][Var] for var:"
                 << (value ? value.defining_op() : nullptr);
        for (const auto& op : operation_set) {
          VLOG(10) << " op: " << op << ",";
        }
        VLOG(10);
        const auto& perm =
            ((src_set.count(node) > 0) ? layout_to_perm("NCHW", "NHWC")
                                       : layout_to_perm("NHWC", "NCHW"));
        const auto& new_layout =
            ((src_set.count(node) > 0) ? common::DataLayout::NHWC
                                       : common::DataLayout::NCHW);
        builder.SetInsertionPointAfter(value.defining_op());
        num_of_transpose_ops++;
        auto transpose_op =
            builder.Build<paddle::dialect::TransposeOp>(value, perm);
        transpose_op->set_attribute(
            "source",
            pir::StrAttribute::get(transpose_op->ir_context(),
                                   "transfer_layout_pass"));
        auto replace_uses_in_cut_set = [&](pir::OpOperand arg) {
          bool is_arg_in_cut_set =
              operation_set.find(arg.owner()) != operation_set.end();
          auto cur_op = arg.owner();
          auto parent = cur_op->GetParentOp();
          while (parent && !is_arg_in_cut_set) {
            is_arg_in_cut_set =
                operation_set.find(parent) != operation_set.end();
            VLOG(10) << "[replace_uses_in_cut_set]" << parent << " "
                     << is_arg_in_cut_set;
            cur_op = parent;
            parent = cur_op->GetParentOp();
          }
          return is_arg_in_cut_set && (arg.owner() != transpose_op.operation());
        };
        pir::SetNewLayoutForValue(transpose_op.out(), new_layout);
        value.ReplaceUsesWithIf(transpose_op.out(), replace_uses_in_cut_set);
      }
    }
    AddStatistics(num_of_transpose_ops, num_of_layout_changed_ops);
  }
};

namespace pir {

std::unique_ptr<pir::Pass> CreateTransferLayoutPass() {
  return std::make_unique<TransferLayoutPass>();
}

}  // namespace pir

REGISTER_IR_PASS(transfer_layout_pass, TransferLayoutPass);
