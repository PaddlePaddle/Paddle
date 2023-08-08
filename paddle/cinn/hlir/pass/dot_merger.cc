// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/pass/infershape.h"

namespace cinn {
namespace hlir {
namespace pass {
namespace {

using common::GraphNode;
using framework::Node;
using framework::NodeData;
using framework::Operator;

template <typename T>
using OpValueType = cinn::hlir::framework::OpValueType<T>;
using infershape_t = std::function<std::vector<framework::shape_t>(
    const std::vector<framework::shape_t>&, const framework::AttrMapType&)>;
using inferdtype_t = std::function<std::vector<Type>(
    const std::vector<Type>&, const framework::AttrMapType&)>;
using dtype_dict_t = absl::flat_hash_map<std::string, common::Type>;
using shape_dict_t = absl::flat_hash_map<std::string, framework::shape_t>;

bool accessible(GraphNode* start, GraphNode* end) {
  std::set<GraphNode const*> marked;
  std::function<void(GraphNode const*)> dfs = [&](GraphNode const* node) {
    marked.emplace(node);
    for (const auto& edge : node->outlinks()) {
      if (!marked.count(edge->sink())) {
        dfs(edge->sink());
      }
    }
  };
  dfs(start);
  return marked.count(end);
}

template <typename T>
T get_attr(Node* instr, const std::string& attr, T def) {
  if (!instr->attrs.attr_store.count(attr)) {
    return def;
  }
  return absl::get<T>(instr->attrs.attr_store.at(attr));
}

NodeData* input_operand(Node* instr, int idx) {
  return instr->inlinks_in_order()[idx]->source()->safe_as<NodeData>();
}
NodeData* output_operand(Node* instr, int idx) {
  return instr->outlinks_in_order()[idx]->sink()->safe_as<NodeData>();
}

void remove_node(framework::Graph* graph, GraphNode* node) {
  auto inlinks = node->inlinks();
  for (auto& link : inlinks) {
    link->source()->UnLinkSingleTo(link->sink());
  }
  auto outlinks = node->outlinks();
  for (auto& link : outlinks) {
    link->source()->UnLinkSingleTo(link->sink());
  }
  graph->DropNode(node);
}

template <typename T>
bool all_equal(const T& arg) {
  return arg[0] == arg[1];
}

template <typename T, typename... Args>
bool all_equal(const T& arg, const Args&... args) {
  return all_equal(arg) && all_equal(args...);
}

void PrintAllMatmulOps(framework::Graph* graph, const std::string& dot_type) {
  auto& dtype_dict{graph->GetMutableAttrs<dtype_dict_t>("inferdtype")};
  auto& shape_dict{graph->GetMutableAttrs<shape_dict_t>("infershape")};
  auto nodes = std::get<0>(graph->topological_order());
  auto print_shape = [](const std::vector<int32_t>& shape) -> std::string {
    std::stringstream ss;
    for (auto i : shape) {
      ss << i << ",";
    }
    return ss.str();
  };
  for (auto* n : nodes) {
    auto* op_node = n->safe_as<Node>();
    if (op_node && op_node->op()->name == dot_type) {
      auto a_id = input_operand(op_node, 0)->id();
      auto b_id = input_operand(op_node, 1)->id();
      auto a_shape = shape_dict.at(a_id);
      auto b_shape = shape_dict.at(b_id);
      LOG(INFO) << "Find op: " << dot_type;
      LOG(INFO) << "Attrs: "
                << "trans_a = " << get_attr<bool>(op_node, "trans_a", false)
                << ", "
                << "trans_b = " << get_attr<bool>(op_node, "trans_b", false)
                << ", "
                << "a: " << a_id << ", " << print_shape(a_shape) << " "
                << "b: " << b_id << ", " << print_shape(b_shape);
    }
  }
}

class DotBuilder {
 public:
  explicit DotBuilder(framework::Graph* graph, std::string dot_type)
      : graph_{graph},
        dot_type_{std::move(dot_type)},
        dtype_dict_{graph_->GetMutableAttrs<dtype_dict_t>("inferdtype")},
        shape_dict_{graph_->GetMutableAttrs<shape_dict_t>("infershape")} {}

  framework::Graph* graph() const { return graph_; }
  const dtype_dict_t& dtype_dict() const { return dtype_dict_; }
  const shape_dict_t& shape_dict() const { return shape_dict_; }

  // Currently the constructor of `NodeData` needs to pass in `Shared<Node>`.
  NodeData* Var(common::Shared<Node>& producer) {  // NOLINT
    auto* res = new NodeData(producer, 0, 0, node_name("var"), false);
    graph_->RegisterNode(producer->id(), res);
    graph_->RegisterNode(res->id(), producer.get());
    producer->LinkTo(res);
    InferShape(producer.get(), dtype_dict_, shape_dict_);
    return res;
  }

  NodeData* Concat(int axis, std::vector<NodeData*> inputs) {
    const std::string type{"concat"};
    auto instr = common::Shared<Node>(
        new Node(framework::Operator::Get(type), type, node_name(type)));
    instr->attrs.attr_store["axis"] = axis;
    for (auto* in : inputs) {
      in->LinkTo(instr.get());
    }
    auto* output = Var(instr);
    return output;
  }

  NodeData* Matmul(bool trans_a,
                   bool trans_b,
                   bool trans_out,
                   float alpha,
                   NodeData* lhs,
                   NodeData* rhs) {
    const std::string type{dot_type_};
    auto instr = common::Shared<Node>(
        new Node(framework::Operator::Get(type), type, node_name(type)));
    matmul_ = instr.get();
    instr->attrs.attr_store["trans_a"] = trans_a;
    instr->attrs.attr_store["trans_b"] = trans_b;
    instr->attrs.attr_store["trans_out"] = trans_out;
    instr->attrs.attr_store["alpha"] = alpha;
    lhs->LinkTo(instr.get());
    rhs->LinkTo(instr.get());
    auto* output = Var(instr);
    return output;
  }

  NodeData* Slice(std::vector<int> axes,
                  std::vector<int> starts,
                  std::vector<int> ends,
                  NodeData* input,
                  NodeData* output) {
    const std::string type{"slice"};
    auto instr = common::Shared<Node>(
        new Node(framework::Operator::Get(type), type, node_name(type)));
    instr->attrs.attr_store["axes"] = std::move(axes);
    instr->attrs.attr_store["starts"] = std::move(starts);
    instr->attrs.attr_store["ends"] = std::move(ends);
    instr->attrs.attr_store["infer_flags"] = std::vector<int>{};
    instr->attrs.attr_store["strides"] = std::vector<int>{};
    instr->attrs.attr_store["decrease_axis"] = std::vector<int>{};
    input->LinkTo(instr.get());
    instr->LinkTo(output);
    graph_->RegisterNode(instr->id(), instr.get());
    InferShape(instr.get(), dtype_dict_, shape_dict_);
    output->source_node = instr;
    return output;
  }

  std::string node_name(std::string prefix) const {
    return std::move(
        prefix.append("__dot_merger_").append(std::to_string(idx_++)));
  }

  Node* matmul_op() const { return matmul_; }

 private:
  static int idx_;
  framework::Graph* graph_{};
  const std::string dot_type_;
  dtype_dict_t& dtype_dict_;
  shape_dict_t& shape_dict_;
  Node* matmul_{};
};

int DotBuilder::idx_ = 0;

class DotMergerPass {
 public:
  // Find the same input for matrix multiplication and recursively fuse.
  static int Apply(framework::Graph* graph, const std::string& dot_type) {
    int cnt{};
    // In the return map, the key is a shared variable, and the values
    // are the dot operators to be fused.
    auto clusters = GetClusters(graph, dot_type);
    std::set<Node*> nodes_to_remove;
    DotBuilder builder(graph, dot_type);
    for (auto& c : clusters) {
      auto& dots = c.second;
      for (size_t i = 0; i < dots.size(); ++i) {
        auto*& a = dots[i];
        if (!a) {
          VLOG(5) << "The node has been fused and removed, skipped.";
          continue;
        }
        std::vector<Node*> merge_nodes;
        merge_nodes.clear();
        merge_nodes.push_back(a);
        for (size_t j = i + 1; j < dots.size(); ++j) {
          auto* b = dots[j];
          if (!b || nodes_to_remove.count(a) || nodes_to_remove.count(b) ||
              accessible(a, b) || accessible(b, a)) {
            VLOG(5) << "Because nodes `" << a->id() << "` and `" << b->id()
                    << " have data dependencies or have been deleted, they "
                       "cannot be merged.";
            continue;
          }
          if (!is_merge(&builder, a, b)) {
            continue;
          }
          merge_nodes.push_back(dots[j]);
        }
        if (merge_nodes.size() < 2) {
          continue;
        }
        auto* merged = NewMergeDots(&builder, merge_nodes);
        cnt += 1;
        for (size_t j = 0; j < merge_nodes.size(); ++j) {
          nodes_to_remove.insert(dots[j]);
          if (j != 0) {
            dots[j] = nullptr;
          }
        }
        dots[i] = merged;
      }
    }

    for (auto* n : nodes_to_remove) {
      remove_node(graph, n);
    }
    return cnt;
  }

 private:
  static std::map<NodeData*, std::vector<Node*>> GetClusters(
      framework::Graph* graph, const std::string& op_type) {
    std::map<NodeData*, std::vector<Node*>> clusters;
    auto nodes = std::get<0>(graph->topological_order());
    for (auto* n : nodes) {
      auto* op_node = n->safe_as<Node>();
      if (op_node && op_node->op()->name == op_type) {
        for (auto& edge : n->inlinks()) {
          auto* var_node = edge->source()->safe_as<NodeData>();
          CHECK(var_node) << "The variable node can not be null.";
          clusters[var_node].push_back(op_node);
        }
      }
    }
    std::vector<std::map<NodeData*, std::vector<Node*>>::iterator> del;
    for (auto it = clusters.begin(); it != clusters.end(); ++it) {
      // At least 2 operators are required to fuse.
      if (it->second.size() < 2) {
        del.push_back(it);
      }
    }
    for (auto& it : del) {
      clusters.erase(it);
    }
    VLOG(3) << "clusters size = " << clusters.size();
    return clusters;
  }

  static bool is_merge(DotBuilder* builder, Node* a, Node* b) {
    CHECK(a && b) << "The pointer of node is illegal.";
    const std::array<bool, 2> trans_a{get_attr<bool>(a, "trans_a", false),
                                      get_attr<bool>(b, "trans_a", false)};
    const std::array<bool, 2> trans_b{get_attr<bool>(a, "trans_b", false),
                                      get_attr<bool>(b, "trans_b", false)};
    const std::array<bool, 2> trans_out{get_attr<bool>(a, "trans_out", false),
                                        get_attr<bool>(b, "trans_out", false)};
    const std::array<float, 2> alpha{get_attr<float>(a, "alpha", 1.f),
                                     get_attr<float>(b, "alpha", 1.f)};
    if (!all_equal(trans_a, trans_b, trans_out, alpha)) {
      return false;
    }
    NodeData *shared_input{}, *input_a{}, *input_b{};
    if (input_operand(a, 1) == input_operand(b, 1)) {
      shared_input = input_operand(a, 1);
      input_a = input_operand(a, 0);
      input_b = input_operand(b, 0);
    } else if (input_operand(a, 0) == input_operand(b, 0)) {
      shared_input = input_operand(a, 0);
      input_a = input_operand(a, 1);
      input_b = input_operand(b, 1);
    } else {
      return false;
    }
    auto* output_a = output_operand(a, 0);
    auto* output_b = output_operand(b, 0);
    auto& graph_outs = builder->graph()->outputs;
    for (auto* n : {shared_input, input_a, input_b}) {
      if (std::find(graph_outs.begin(), graph_outs.end(), n) !=
          graph_outs.end()) {
        return false;
      }
    }
    return true;
  }

  static Node* NewMergeDots(DotBuilder* builder,
                            std::vector<Node*> merge_nodes) {
    const std::array<bool, 2> trans_a{
        get_attr<bool>(merge_nodes[0], "trans_a", false),
        get_attr<bool>(merge_nodes[1], "trans_a", false)};
    const std::array<bool, 2> trans_b{
        get_attr<bool>(merge_nodes[0], "trans_b", false),
        get_attr<bool>(merge_nodes[1], "trans_b", false)};
    const std::array<float, 2> alpha{
        get_attr<float>(merge_nodes[0], "alpha", 1.f),
        get_attr<float>(merge_nodes[1], "alpha", 1.f)};

    bool lhs{true};
    int axis{1};
    NodeData* shared_input = input_operand(merge_nodes[0], 0);

    if (input_operand(merge_nodes[0], 1) == input_operand(merge_nodes[1], 1)) {
      shared_input = input_operand(merge_nodes[0], 1);
      lhs = false;
      if (!trans_a[0]) {
        axis = 0;
      } else if (trans_b[0]) {
        axis = 0;
      }
    }
    CHECK(shared_input) << "The input node type must be variable.";
    std::vector<NodeData*> concat_nodes;
    concat_nodes.clear();
    auto shape_shared = builder->shape_dict().at(shared_input->id());
    concat_nodes.push_back(input_operand(merge_nodes[0], axis));
    for (size_t i = 1; i < merge_nodes.size(); ++i) {
      auto shape_a = builder->shape_dict().at(
          input_operand(merge_nodes[i - 1], axis)->id());
      auto shape_b =
          builder->shape_dict().at(input_operand(merge_nodes[i], axis)->id());
      CHECK_EQ(shape_a[1 - axis], shape_b[1 - axis])
          << "The shape of matmul is error. " << shape_a.size() << ", "
          << shape_b.size();
      concat_nodes.push_back(input_operand(merge_nodes[i], axis));
    }
    auto* concat_out = builder->Concat(axis, concat_nodes);
    NodeData* matmul_out{};
    if (!lhs) {
      matmul_out = builder->Matmul(
          trans_a[0], trans_b[0], false, alpha[0], concat_out, shared_input);
    } else {
      matmul_out = builder->Matmul(
          trans_a[0], trans_b[0], false, alpha[0], shared_input, concat_out);
    }
    auto start_shape = 0;
    for (size_t i = 0; i < concat_nodes.size(); ++i) {
      auto shape =
          builder->shape_dict().at(input_operand(merge_nodes[i], axis)->id());
      auto* output = output_operand(merge_nodes[i], 0);
      builder->Slice({axis},
                     {start_shape},
                     {start_shape + shape[axis]},
                     matmul_out,
                     output);
      start_shape += shape[axis];
    }
    return builder->matmul_op();
  }

  static Node* MergeDots(DotBuilder* builder, Node* a, Node* b) {
    CHECK(a && b) << "The pointer of node is illegal.";
    const std::array<bool, 2> trans_a{get_attr<bool>(a, "trans_a", false),
                                      get_attr<bool>(b, "trans_a", false)};
    const std::array<bool, 2> trans_b{get_attr<bool>(a, "trans_b", false),
                                      get_attr<bool>(b, "trans_b", false)};
    const std::array<bool, 2> trans_out{get_attr<bool>(a, "trans_out", false),
                                        get_attr<bool>(b, "trans_out", false)};
    const std::array<float, 2> alpha{get_attr<float>(a, "alpha", 1.f),
                                     get_attr<float>(b, "alpha", 1.f)};
    if (!all_equal(trans_a, trans_b, trans_out, alpha)) {
      return nullptr;
    }
    bool lhs{true};
    int axis{1};
    NodeData *shared_input{}, *input_a{}, *input_b{};
    if (input_operand(a, 1) == input_operand(b, 1)) {
      shared_input = input_operand(a, 1);
      input_a = input_operand(a, 0);
      input_b = input_operand(b, 0);
      lhs = false;
      if (!trans_a[0]) {
        axis = 0;
      } else if (trans_b[0]) {
        axis = 0;
      }
    } else if (input_operand(a, 0) == input_operand(b, 0)) {
      shared_input = input_operand(a, 0);
      input_a = input_operand(a, 1);
      input_b = input_operand(b, 1);
    } else {
      return nullptr;
    }
    auto* output_a = output_operand(a, 0);
    auto* output_b = output_operand(b, 0);
    auto& graph_outs = builder->graph()->outputs;
    for (auto* n : {shared_input, input_a, input_b}) {
      if (std::find(graph_outs.begin(), graph_outs.end(), n) !=
          graph_outs.end()) {
        return nullptr;
      }
    }
    CHECK(shared_input && input_a && input_b)
        << "The input node type must be variable.";
    auto shape_shared = builder->shape_dict().at(shared_input->id());
    auto shape_a = builder->shape_dict().at(input_a->id());
    auto shape_b = builder->shape_dict().at(input_b->id());
    CHECK_EQ(shape_a[1 - axis], shape_b[1 - axis])
        << "The shape of matmul is error. " << shape_a.size() << ", "
        << shape_b.size();
    auto* concat_out = builder->Concat(axis, {input_a, input_b});
    NodeData* matmul_out{};
    if (!lhs) {
      matmul_out = builder->Matmul(
          trans_a[0], trans_b[0], false, alpha[0], concat_out, shared_input);
    } else {
      matmul_out = builder->Matmul(
          trans_a[0], trans_b[0], false, alpha[0], shared_input, concat_out);
    }
    builder->Slice({axis}, {0}, {shape_a[axis]}, matmul_out, output_a);
    builder->Slice({axis},
                   {shape_a[axis]},
                   {shape_a[axis] + shape_b[axis]},
                   matmul_out,
                   output_b);
    return builder->matmul_op();
  }
};

}  // namespace

void DotMergerPassFunc(framework::Graph* graph) {
  // The cublas gemm is not yet supported.
  for (auto& dot_type : {"matmul", "cublas_matmul"}) {
    int n = DotMergerPass::Apply(graph, dot_type);
    VLOG(3) << "The fusion of `" << dot_type << "` was performed " << n
            << " times.";
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(DotMerger) {
  CINN_REGISTER_PASS(DotMerger)
      .describe("")
      .set_change_structure(false)
      .provide_graph_attr("infershape")
      .provide_graph_attr("inferdtype")
      .set_body(cinn::hlir::pass::DotMergerPassFunc);
  return true;
}
