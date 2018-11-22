// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/ir_passes/mem_optim_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_traits.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Graph;
using framework::ir::Node;

const int kBatchSize = 13;  // replace the placement -1 in shape

std::unique_ptr<framework::ir::Graph> MemOptimPass::ApplyImpl(
    std::unique_ptr<framework::ir::Graph> graph) const {
  graph_ = graph.get();
  std::unordered_map<std::string, std::string> reuse_table;
  MakeReusePlan(&reuse_table);
  PerformReusePlan(reuse_table);
  return graph;
}

// Collect the lifecycles of the tensors.
// Traverse the graph in topological order.
// TODO(Superjomn) the traversal order also affect the lifecycles, try to change
// the order to improve the reuse performance latter.
void MemOptimPass::CollectLifeCycle(
    std::unordered_map<std::string, lifecycle_t>* lifecycles) const {
  max_lifecycle_ = 0;
  for (auto* op_node : framework::ir::TopologyDfsSortOperations(*graph_)) {
    if (!op_node->IsOp()) continue;
    auto reads = op_node->inputs;
    auto writes = op_node->outputs;

    std::vector<Node*> requires(reads.begin(), reads.end());
    requires.insert(requires.end(), writes.begin(), writes.end());

    bool to_optim = true;

    if (op_node->Name() == "stack") to_optim = false;

    // Disable reuse of feed variables.
    if (op_node->Name() == "feed") {
      for (auto* node : op_node->outputs) {
        auto var = node->Name();
        if (!lifecycles->count(var)) {
          (*lifecycles)[var] =
              std::make_pair(max_lifecycle_, std::numeric_limits<int>::max());
        }
      }
    } else {
      // Normal operators.
      for (const Node* node : requires) {
        if (node->Var()->Persistable()) continue;
        std::string var = node->Name();
        if (!lifecycles->count(var)) {
          int dead = to_optim ? max_lifecycle_ : std::numeric_limits<int>::max();
          (*lifecycles)[var] = std::make_pair(max_lifecycle_, dead);
        } else {
          (*lifecycles)[var].second =
              std::max(max_lifecycle_, lifecycles->at(var).second);  // max()
        }
      }
    }

    ++max_lifecycle_;
  }
}

// TODO(Superjomn) Make this a general help method.
int DataTypeToSpace(framework::proto::VarType_Type type) {
  switch (type) {
    case framework::proto::VarType_Type_BOOL:
      return sizeof(bool);
    case framework::proto::VarType_Type_FP32:
      return sizeof(float);
    case framework::proto::VarType_Type_INT32:
      return sizeof(int32_t);
    case framework::proto::VarType_Type_INT64:
      return sizeof(int64_t);
    default:
      PADDLE_THROW("Unknown data type");
  }
}

// A tensor's shape with its data type to memory size.
int ShapeToSpace(const std::vector<long>& shape,
                 framework::proto::VarType_Type data_type) {
  auto total_dim =
      std::accumulate(shape.begin(), shape.end(), 1, [](long a, long b) {
        if (a == -1) a = kBatchSize;
        return a * b;
      });
  int data_type_space = DataTypeToSpace(data_type);
  PADDLE_ENFORCE_GT(data_type_space, 0);
  int space = total_dim * data_type_space;
  return space;
}

// Collect the shape information of the tensors.
void MemOptimPass::CollectShapes(
    std::unordered_map<std::string, Node*>* tensor_nodes,
    std::unordered_map<std::string, int>* space_table) const {
  // Collect tensors from graph.
  for (auto* node : graph_->Nodes()) {
    if (node->IsVar() &&
        node->Var()->GetType() ==
            framework::proto::VarType::Type::VarType_Type_LOD_TENSOR) {
      // Parameters will not be reused.
      if (node->Var()->Persistable()) continue;
      (*tensor_nodes)[node->Name()] = node;
      (*space_table)[node->Name()] =
          ShapeToSpace(node->Var()->GetShape(), node->Var()->GetDataType());
    }
  }
}

// Find a sutable (big enough but smallest to avoid memory waste).
//
// Args:
// @tensor_nodes: the tensor nodes in the ir::Graph.
// @free_existing_tensors: the allocated tensor and are free.
// @space_table: the memory space of tensors.
// @tensor2use: the tensor that requires memory.
//
// Returns:
// true if found some existing tensor to reuse.
// false if no sutable tensor to reuse, one need to allocate a new tensor for
// this requirement.
__attribute__((warn_unused_result)) //
bool FindSutableTensorToReuse(
    int space_required,
    const std::unordered_map<std::string, Node*>& tensor_nodes,
    std::unordered_set<std::string>* free_existing_tensors,
    const std::unordered_map<std::string, int> &space_table,
    std::string* tensor2use) {
  std::pair<std::string, int> best_fit;
  best_fit.second = std::numeric_limits<int>::max();

  for (auto& tensor : *free_existing_tensors) {
    if (!space_table.count(tensor)) continue;
    int space = space_table.at(tensor);
    int space_diff = space - space_required;
    if (space_diff >= 0 && space_diff < best_fit.second) {
      best_fit.first = tensor;
      best_fit.second = space_diff;
    }
  }

  if (best_fit.second < std::numeric_limits<int>::max()) {
    *tensor2use = best_fit.first;
    return true;
  }
  return false;
}

// Allocate new tensor instead of reusing the existing one.
void AllocateNewTensor(
    const std::string& name, int space_required,
    const std::unordered_map<std::string, Node*>& tensor_nodes,
    std::unordered_set<std::string>* free_existing_tensors,
    std::unordered_map<std::string, int>* space_table,
    std::unordered_map<std::string, std::string>* reuse_table) {
  // The newly born tensor is free to be used.
  free_existing_tensors->insert(name);
  // Register the space it has.
  space_table->emplace(name, space_required);
  // The allocated new tensor use the memory of itself.
  (*reuse_table)[name] = name;
}

// Free a tensor and make it resuable.
// @tensor: the tensor to free.
// @free_existing_tensors: the free and allocated tensors.
// @reuse_table: a map from a fake tensor to the existing allocated tensor.
void FreeATensor(const std::string& tensor,
                 std::unordered_set<std::string>* free_existing_tensors,
                 std::unordered_map<std::string, std::string>* reuse_table) {
  if (tensor == "feed" || tensor == "fetch") return;
  // the really allocated tensor.
  const auto& free_tensor = reuse_table->at(tensor);

  free_existing_tensors->insert(free_tensor);
}

// Reuse a free existing tensor.
void ReuseATensor(const std::string& tensor, const std::string& tensor2reuse,
                  std::unordered_set<std::string>* free_existing_tensors,
                  std::unordered_map<std::string, std::string>* reuse_table) {
  auto it = free_existing_tensors->find(tensor2reuse);
  PADDLE_ENFORCE(it != free_existing_tensors->end());
  free_existing_tensors->erase(it);
  (*reuse_table)[tensor] = tensor2reuse;
}

void MemoryStatis(
    const std::unordered_map<std::string, std::string>& reuse_table,
    const std::unordered_map<std::string, int>& space_table,
    long long int* allocated, long long int* saved) {
  *allocated = 0;
  *saved = 0;
  for (auto elem : reuse_table) {
    // if (elem.first == "feed" || elem.second == "second") continue;
    if (elem.first == elem.second) {
      *allocated += space_table.at(elem.first);
    } else {
      *saved += space_table.at(elem.first);
      LOG(INFO) << "reuse " << elem.first << " -> " << elem.second;
    }
  }
}

void MemOptimPass::MakeReusePlan(
    std::unordered_map<std::string, std::string>* reuse_table) const {
  std::unordered_map<std::string, lifecycle_t> lifecycles;
  std::unordered_map<std::string, Node*> tensor_nodes;
  // The allocated tensors whose memory can be reused, they will live across the
  // program execution.
  std::unordered_set<std::string> existing_tensors;
  // The existing tensor that has been allocated, and is also free to reuse.
  std::unordered_set<std::string> free_existing_tensors;
  std::unordered_map<std::string, int> space_table;

  CollectLifeCycle(&lifecycles);
  CollectShapes(&tensor_nodes, &space_table);

  for (int age = 0; age < max_lifecycle_; ++age) {
    std::unordered_set<std::string> born_tensors;
    std::unordered_set<std::string> dead_tensors;
    // Gather the dead and born tensors.
    for (auto elem_it = lifecycles.begin(); elem_it != lifecycles.end();
         elem_it++) {
      if (elem_it->second.first == -1) continue;
      const auto& tensor = elem_it->first;
      const auto& lifecycle = elem_it->second;
      VLOG(4) << "process " << tensor << " reuse " << lifecycle.first << "->"
              << lifecycle.second;

      // Collect newly born tensors.
      if (lifecycle.first == age) {
        born_tensors.insert(tensor);
      }
      // Collect dead tensors whose memory can be reused.
      else if (lifecycle.second < age) {
        dead_tensors.insert(tensor);
        // remove to avoid duplicate process.
        elem_it->second.first = -1;  // avoid duplicate search
      }
    }

    // Reuse the dead tensors for born tensors
    for (const auto& tensor : born_tensors) {
      // Skip the feed and fetch tensor for that they share data with others.
      std::string tensor2reuse;
      if (!space_table.count(tensor)) continue;
      int space_required = space_table.at(tensor);
      if (FindSutableTensorToReuse(space_required, tensor_nodes,
                                   &free_existing_tensors, space_table,
                                   &tensor2reuse)) {
        if (tensor != tensor2reuse) {
          LOG(INFO) << tensor << " -> " << tensor2reuse;
        }
        ReuseATensor(tensor, tensor2reuse, &free_existing_tensors, reuse_table);
      } else {
        VLOG(4) << "allocate " << tensor;
        AllocateNewTensor(tensor, space_required, tensor_nodes,
                          &free_existing_tensors, &space_table, reuse_table);
        ReuseATensor(tensor, tensor, &free_existing_tensors, reuse_table);
      }
    }

    for (const auto& tensor : dead_tensors) {
      // free its memory.
      FreeATensor(tensor, &free_existing_tensors, reuse_table);
    }
  }

  long long allocated, saved_memory;
  MemoryStatis(*reuse_table, space_table, &allocated, &saved_memory);
  LOG(INFO) << "Allocated " << allocated / 1024. / 1024. << " MB for workspace";
  LOG(INFO) << "Saved " << saved_memory / 1024. / 1024. << " MB";
  LOG(INFO) << "The saving ratio: "
            << static_cast<float>(saved_memory) / (saved_memory + allocated);
}

void BuildVarNodeTable(Graph* graph,
                       std::unordered_map<std::string, Node*>* var_node_table) {
  for (auto* node : graph->Nodes()) {
    if (node->IsVar()) {
      (*var_node_table)[node->Name()] = node;
    }
  }
}

void UpdateIrGraphByReuse(
    Graph* graph,
    const std::unordered_map<std::string, std::string>& reuse_table,
    const std::unordered_map<std::string, Node*>& var_node_table) {
  // Unneeded nodes.
  std::unordered_set<const Node*> nodes2rm;
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    for (auto*& x : node->inputs) {
      PADDLE_ENFORCE(x->IsVar());
      auto name = x->Var()->Name();
      if (reuse_table.count(name) && reuse_table.at(name) != name) {
        nodes2rm.insert(var_node_table.at(name));
        auto* node = var_node_table.at(reuse_table.at(name));
        x = node;
      }
    }

    for (auto*& x : node->outputs) {
      PADDLE_ENFORCE(x->IsVar());
      auto name = x->Var()->Name();
      if (reuse_table.count(name) && reuse_table.at(name) != name) {
        nodes2rm.insert(var_node_table.at(name));
        auto* node = var_node_table.at(reuse_table.at(name));
        x = node;
      }
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2rm);
}

void UpdateOpDescsByReuse(
    Graph* graph,
    const std::unordered_map<std::string, std::string>& reuse_table) {
  // for (auto* node : framework::ir::TopologyDfsSortOperations(*graph)) {
  for (auto* node : framework::ir::TopologySortOperations(*graph)) {
    if (node->IsOp()) {
      // Replace the original inputs/outputs with the reused tensors.
      std::unordered_map<std::string, std::vector<std::string>> in_args,
          out_args;
      for (auto argument : node->Op()->Inputs()) {
        for (auto x : argument.second) {
          auto name = x;
          if (reuse_table.count(x) && reuse_table.at(x) != x) {
            name = reuse_table.at(x);
          }
          in_args[argument.first].push_back(name);
          VLOG(4) << node->Name() << " input " << x << " -> " << name;
        }
      }

      for (auto argument : node->Op()->Outputs()) {
        for (auto x : argument.second) {
          auto name = x;
          if (reuse_table.count(x) && reuse_table.at(x) != x) {
            name = reuse_table.at(x);
          }
          out_args[argument.first].push_back(name);
          VLOG(4) << node->Name() << " output " << x << " -> " << name;
        }
      }

      // Update arguments.
      for (auto& arg : in_args) {
        node->Op()->SetInput(arg.first, arg.second);
      }
      for (auto& arg : out_args) {
        node->Op()->SetOutput(arg.first, arg.second);
      }
      node->Op()->Flush();
    }
  }
}

void MemOptimPass::PerformReusePlan(
    const std::unordered_map<std::string, std::string>& reuse_table) const {
  std::unordered_map<std::string, Node*> var_node_table;
  BuildVarNodeTable(graph_, &var_node_table);
  // UpdateIrGraphByReuse(graph_, reuse_table, var_node_table);
  UpdateOpDescsByReuse(graph_, reuse_table);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(memory_optim_pass, paddle::inference::analysis::MemOptimPass);
