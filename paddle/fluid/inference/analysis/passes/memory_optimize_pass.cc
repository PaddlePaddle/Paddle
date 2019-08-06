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

#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include <algorithm>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_to_program_pass.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Graph;
using framework::ir::Node;
using framework::ir::TopologyVarientSort;
using space_table_t = MemoryOptimizePass::space_table_t;

typedef struct {
  std::string name;
  size_t size;
  int cluster;
  std::pair<int, int> lifetime;
  std::unordered_set<std::string> adj;
} MemNode;

// Collect the lifecycles of the tensors.
// Traverse the graph in topological order.
// The traversal order also affect the lifecycles, so different sort_kind is
// used.
void MemoryOptimizePass::CollectLifeCycle(
    std::unordered_map<std::string, lifecycle_t>* lifecycles,
    int sort_kind) const {
  max_lifecycle_ = 0;
  for (auto* op_node : framework::ir::TopologyVarientSort(
           *graph_, static_cast<framework::ir::SortKind>(sort_kind))) {
    if (!op_node->IsOp()) continue;
    auto reads = op_node->inputs;
    auto writes = op_node->outputs;

    std::vector<Node*> requires(reads.begin(), reads.end());
    requires.insert(requires.end(), writes.begin(), writes.end());

    // Disable reuse of feed variables.
    if (op_node->Name() == "feed") {
      for (auto* node : op_node->outputs) {
        auto var = node->Name();
        lifecycles->emplace(var,
                            std::make_pair(0, std::numeric_limits<int>::max()));
      }
    } else {
      // Normal operators.
      for (const Node* node : requires) {
        if (node->Var()->Persistable()) continue;
        std::string var = node->Name();
        if (!lifecycles->count(var)) {
          (*lifecycles)[var] = std::make_pair(max_lifecycle_, max_lifecycle_);
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

void MemoryOptimizePass::CollectVarMemorySize(
    space_table_t* space_table) const {
  const int fake_batch_size = 1;
  // Collect tensors from graph.
  for (auto* node : graph_->Nodes()) {
    if (node->IsVar() &&
        node->Var()->GetType() ==
            framework::proto::VarType::Type::VarType_Type_LOD_TENSOR) {
      // Parameters will not be reused.
      if (node->Var()->Persistable()) continue;
      auto shape = node->Var()->GetShape();
      for (auto& v : shape) {
        if (v < 0) v = fake_batch_size;
      }

      int size = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<int>());
      (*space_table)[node->Var()->Name()] =
          size * DataTypeToSpace(node->Var()->GetDataType());
    }
  }
}

void MakeSimpleReusePlan(
    const std::unordered_map<std::string, std::pair<int, int>>& lifecycles,
    const std::unordered_map<std::string, size_t>& space_table,
    std::unordered_map<std::string, std::string>* node2cluster,
    std::unordered_map<std::string, int>* cluster_size) {
  std::vector<MemNode> mem_nodes;
  for (auto& data : lifecycles) {
    MemNode temp_node;
    temp_node.name = data.first;
    PADDLE_ENFORCE(
        space_table.count(data.first),
        "%s variable should be in the spacetable during memory optimize",
        data.first);
    temp_node.size = space_table.at(data.first);
    temp_node.cluster = -1;
    temp_node.lifetime = data.second;
    mem_nodes.push_back(temp_node);
  }
  auto overlap = [](std::pair<int, int> a, std::pair<int, int> b) -> bool {
    return b.second >= a.first && a.second >= b.first;
  };
  // If the lifetime of two nodes is overwritten, we set them as adjacent nodes.
  for (size_t i = 0; i < mem_nodes.size(); i++) {
    for (size_t j = i + 1; j < mem_nodes.size(); j++) {
      if (overlap(mem_nodes[i].lifetime, mem_nodes[j].lifetime)) {
        mem_nodes[i].adj.insert(mem_nodes[j].name);
        mem_nodes[j].adj.insert(mem_nodes[i].name);
      }
    }
  }

  // Sort the nodes according to the node memory size.
  auto sort_func = [](MemNode a, MemNode b) { return a.size > b.size; };
  std::sort(mem_nodes.begin(), mem_nodes.end(), sort_func);

  // Generating Memory Reuse Strategy Based on Greedy Way
  for (size_t i = 0; i < mem_nodes.size(); i++) {
    if (mem_nodes[i].cluster >= 0) continue;
    int cluster_index = cluster_size->size();
    mem_nodes[i].cluster = cluster_index;
    (*cluster_size)[mem_nodes[i].name] = mem_nodes[i].size;
    (*node2cluster)[mem_nodes[i].name] = mem_nodes[i].name;
    std::unordered_set<std::string> cluster_adj = mem_nodes[i].adj;
    for (size_t j = i + 1; j < mem_nodes.size(); j++) {
      if (mem_nodes[j].cluster < 0 &&
          (cluster_adj.find(mem_nodes[j].name) == cluster_adj.end())) {
        (*node2cluster)[mem_nodes[j].name] = mem_nodes[i].name;
        mem_nodes[j].cluster = cluster_index;
        for (auto& n : mem_nodes[j].adj) {
          cluster_adj.insert(n);
        }
      }
    }
  }
  for (auto& cluster : *cluster_size) {
    LOG(INFO) << "Cluster name : " << cluster.first
              << "  size: " << cluster.second;
  }
}

// Collect the memory size of the tensors.
void MemoryOptimizePass::CollectVarMemorySize(
    const std::unordered_map<std::string, size_t>& batch_var_ave_dim,
    std::unordered_map<std::string, Node*>* tensor_nodes,
    space_table_t* space_table) const {
  // Collect tensors from graph.
  for (auto* node : graph_->Nodes()) {
    if (node->IsVar() &&
        node->Var()->GetType() ==
            framework::proto::VarType::Type::VarType_Type_LOD_TENSOR) {
      // Parameters will not be reused.
      if (node->Var()->Persistable()) continue;
      (*tensor_nodes)[node->Name()] = node;
      (*space_table)[node->Name()] =
          DataTypeToSpace(node->Var()->GetDataType()) *
          batch_var_ave_dim.at(node->Name());
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
// The suitable tensor for reuse is one that is approximately equal to the
// memory demand.
bool FindSuitableTensorToReuse(
    const std::string& tensor, int space_required,
    const std::unordered_map<std::string, Node*>& tensor_nodes,
    std::unordered_set<std::string>* free_existing_tensors,
    const space_table_t& space_table,
    const std::vector<std::unordered_set<std::string>>& var_clusters,
    std::string* tensor2use) __SHOULD_USE_RESULT__;

bool FindSuitableTensorToReuse(
    const std::string& tensor, int space_required,
    const std::unordered_map<std::string, Node*>& tensor_nodes,
    std::unordered_set<std::string>* free_existing_tensors,
    const space_table_t& space_table,
    const std::vector<std::unordered_set<std::string>>& var_clusters,
    std::string* tensor2use) {
  std::pair<std::string, size_t> best_fit;
  best_fit.second = std::numeric_limits<int>::max();
  VLOG(5) << "Split Tensors to " << var_clusters.size() << " clusters";

  // find the cluster this var belongs to.
  const std::unordered_set<std::string>* cluster = nullptr;
  for (const auto& c : var_clusters) {
    if (c.count(tensor)) {
      cluster = &c;
      break;
    }
  }
  PADDLE_ENFORCE_NOT_NULL(cluster,
                          "something wrong in memory optimization, the "
                          "variable %s not in the clusters.",
                          tensor);

  for (auto& candidate : *free_existing_tensors) {
    // This is not a temporary tensor.
    if (!space_table.count(candidate)) continue;
    // Not in the same cluster.
    if (!cluster->count(candidate)) continue;

    size_t space = space_table.at(candidate);
    PADDLE_ENFORCE(
        space <= std::numeric_limits<std::make_signed<size_t>::type>::max(),
        "space overload");
    size_t space_diff =
        std::abs((std::make_signed<size_t>::type)space - space_required);
    if (space_diff < best_fit.second) {
      best_fit.first = candidate;
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
    const std::string& name, size_t space_required,
    const std::unordered_map<std::string, Node*>& tensor_nodes,
    std::unordered_set<std::string>* free_existing_tensors,
    space_table_t* space_table,
    std::unordered_map<std::string, std::string>* reuse_table) {
  // The newly born tensor is free to be used.
  free_existing_tensors->insert(name);
  // Register the space it has.
  PADDLE_ENFORCE(space_table->count(name));
  space_table->at(name) = std::max(space_table->at(name), space_required);
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
                  size_t memory_size,
                  std::unordered_set<std::string>* free_existing_tensors,
                  std::unordered_map<std::string, std::string>* reuse_table,
                  space_table_t* reused_space_table) {
  auto it = free_existing_tensors->find(tensor2reuse);
  PADDLE_ENFORCE(it != free_existing_tensors->end());
  free_existing_tensors->erase(it);
  (*reuse_table)[tensor] = tensor2reuse;
  // Update the memory size of a reused tensor, the memory will grow if the
  // required memory is larger.
  (*reused_space_table)[tensor2reuse] =
      std::max(reused_space_table->at(tensor2reuse), memory_size);
}

// Calculate the memory usage.
void EvaluateMemoryUsage(
    const std::unordered_map<std::string, std::string>& reuse_table,
    const space_table_t& space_table,
    const std::unordered_map<std::string, size_t>& var_batch_ave_size,
    size_t* allocated, size_t* saved) {
  *allocated = 0;
  *saved = 0;

  for (auto elem : reuse_table) {
    if (elem.first == elem.second) {
      *allocated += space_table.at(elem.first);
      VLOG(4) << elem.first << " <-> " << elem.second << " "
              << space_table.at(elem.first) << " "
              << space_table.at(elem.second);
    } else {
      *saved += space_table.at(elem.first);
      VLOG(4) << "reuse " << elem.first << " -> " << elem.second;
    }
  }
  VLOG(4) << "allocated " << *allocated;
  VLOG(4) << "saved " << *saved;
}

// Return saved ratio.
void MemoryOptimizePass::MakeReusePlan(
    const std::vector<std::unordered_set<std::string>>& var_clusters,
    const std::unordered_map<std::string, size_t>& var_batch_ave_size,
    const space_table_t& space_table,
    std::unordered_map<std::string, std::string>* reuse_table, int sort_kind,
    MemoryAllocation* memory_allocation) const {
  // Clear the existing plan.
  reuse_table->clear();

  // The `space_table` stores the real memory size for each tensor.
  // The `reused_space_table` stores the maximum memory size required by a
  // tensor during the memory reusing, the small tensor might be reused by a
  // larger tensor, and the memory size of the small one will grow.
  auto reused_space_table = space_table;

  std::unordered_map<std::string, lifecycle_t> life_cycles;
  std::unordered_map<std::string, Node*> tensor_nodes;
  // The allocated tensors whose memory can be reused, they will live across the
  // program execution.
  std::unordered_set<std::string> existing_tensors;
  // The existing tensor that has been allocated, and is also free to reuse.
  std::unordered_set<std::string> free_existing_tensors;

  CollectLifeCycle(&life_cycles, sort_kind);

  for (int age = 0; age < max_lifecycle_; ++age) {
    std::unordered_set<std::string> born_tensors;
    std::unordered_set<std::string> dead_tensors;
    // Gather the dead and born tensors.
    for (auto elem_it = life_cycles.begin(); elem_it != life_cycles.end();
         elem_it++) {
      if (elem_it->second.first == -1) {
        continue;
      }
      const auto& tensor = elem_it->first;
      const auto& lifecycle = elem_it->second;
      VLOG(4) << "process " << tensor << " reuse " << lifecycle.first << "->"
              << lifecycle.second;

      // Collect newly born tensors.
      if (lifecycle.first == age) {
        born_tensors.insert(tensor);
      }
      // Collect dead tensors whose memory can be reused.
      else if (lifecycle.second < age) {  // NOLINT
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
      size_t space_required = space_table.at(tensor);
      if (FindSuitableTensorToReuse(tensor, space_required, tensor_nodes,
                                    &free_existing_tensors, reused_space_table,
                                    var_clusters, &tensor2reuse)) {
        if (tensor != tensor2reuse) {
          VLOG(4) << tensor << " -> " << tensor2reuse;
        }
        ReuseATensor(tensor, tensor2reuse, space_required,
                     &free_existing_tensors, reuse_table, &reused_space_table);
      } else {
        VLOG(4) << "allocate " << tensor;
        AllocateNewTensor(tensor, space_required, tensor_nodes,
                          &free_existing_tensors, &reused_space_table,
                          reuse_table);
        ReuseATensor(tensor, tensor, space_required, &free_existing_tensors,
                     reuse_table, &reused_space_table);
      }
    }

    for (const auto& tensor : dead_tensors) {
      // free its memory.
      FreeATensor(tensor, &free_existing_tensors, reuse_table);
    }
  }

  EvaluateMemoryUsage(*reuse_table, reused_space_table, var_batch_ave_size,
                      &(memory_allocation->allocated),
                      &(memory_allocation->saved));
  memory_allocation->sort_kind = sort_kind;
}

void BuildVarNodeTable(Graph* graph,
                       std::unordered_map<std::string, Node*>* var_node_table) {
  for (auto* node : graph->Nodes()) {
    if (node->IsVar()) {
      (*var_node_table)[node->Name()] = node;
    }
  }
}

// NOTE The optimized opdesc doesn't match ir::Graph.
void UpdateOpDescsByReuse(
    Graph* graph,
    const std::unordered_map<std::string, std::string>& reuse_table,
    int sort_kind) {
  // TODO(Superjomn) change here to be compatible with the runtime order.
  for (auto* node : TopologyVarientSort(
           *graph, static_cast<framework::ir::SortKind>(sort_kind))) {
    if (node->IsOp()) {
      // Replace the original inputs/outputs with the reused tensors.
      std::unordered_map<std::string, std::vector<std::string>> in_args,
          out_args;
      for (auto argument : node->Op()->Inputs()) {
        for (const auto& x : argument.second) {
          auto name = x;
          if (reuse_table.count(x) && reuse_table.at(x) != x) {
            name = reuse_table.at(x);
          }
          in_args[argument.first].push_back(name);
          VLOG(4) << node->Name() << " input " << x << " -> " << name;
        }
      }

      // modify the graph
      for (auto input_node : node->inputs) {
        PADDLE_ENFORCE(input_node->IsVar());
        std::string input_node_name = input_node->Name();
        if (reuse_table.count(input_node_name) &&
            reuse_table.at(input_node_name) != input_node_name) {
          auto name = reuse_table.at(input_node_name);
          input_node->RenameVar(name);
        }
      }

      for (auto argument : node->Op()->Outputs()) {
        for (const auto& x : argument.second) {
          auto name = x;
          if (reuse_table.count(x) && reuse_table.at(x) != x) {
            name = reuse_table.at(x);
          }
          out_args[argument.first].push_back(name);
          VLOG(4) << node->Name() << " output " << x << " -> " << name;
        }
      }

      // modify the graph
      for (auto out_node : node->outputs) {
        PADDLE_ENFORCE(out_node->IsVar());
        std::string out_node_name = out_node->Name();
        if (reuse_table.count(out_node_name) &&
            reuse_table.at(out_node_name) != out_node_name) {
          auto name = reuse_table.at(out_node_name);
          out_node->RenameVar(name);
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

void MemoryOptimizePass::PerformReusePlan(
    const std::unordered_map<std::string, std::string>& reuse_table,
    int sort_kind, std::unordered_set<std::string>* vars2remove) const {
  std::unordered_map<std::string, Node*> var_node_table;
  BuildVarNodeTable(graph_, &var_node_table);
  UpdateOpDescsByReuse(graph_, reuse_table, sort_kind);

  for (auto& item : reuse_table) {
    if (item.first != item.second) {
      vars2remove->insert(item.first);
    }
  }
  VLOG(2) << "to remove vars " << vars2remove->size();
}

std::vector<std::string> split(const std::string& line, char delim) {
  std::vector<std::string> res;
  std::string field;
  std::stringstream line_stream(line);
  while (std::getline(line_stream, field, delim)) {
    res.emplace_back(field);
  }
  return res;
}

// Deserialize the batch var shapes from the cache file.
std::vector<std::map<std::string, std::vector<int>>> DeseralizeBatchVarShapes(
    const std::string& path) {
  std::ifstream file(path);
  PADDLE_ENFORCE(file.is_open(), "failed to open %s  to read cache", path);
  std::string line;
  std::vector<std::map<std::string, std::vector<int>>> batch_shapes;

  while (std::getline(file, line)) {
    std::map<std::string, std::vector<int>> batch;
    for (const auto& var_info : split(line, ';')) {
      auto fields = split(var_info, ':');
      PADDLE_ENFORCE_EQ(fields.size(), 2UL);
      auto var_name = fields.front();
      auto shape_str = split(fields[1], ',');
      std::vector<int> shape;
      for (const auto& v : shape_str) shape.push_back(std::stoi(v));
      batch[var_name] = shape;
    }
    batch_shapes.push_back(batch);
  }
  return batch_shapes;
}

// Replace the -1 in shape to a real number to fake the shape.
std::vector<std::map<std::string, std::vector<int>>> FakeBatchVarShapes(
    const framework::ProgramDesc& program) {
  std::vector<std::map<std::string, std::vector<int>>> res;
  res.emplace_back();
  auto& record = res.front();
  const int fake_batch_size = 3;
  for (auto* var : program.Block(0).AllVars()) {
    if (var->GetType() ==
        framework::proto::VarType::Type::VarType_Type_LOD_TENSOR) {
      auto shape = var->GetShape();
      for (auto& v : shape) {
        if (v < 0) v = fake_batch_size;
      }
      record[var->Name()].assign(shape.begin(), shape.end());
    }
  }
  return res;
}

// Calculate the average dim of each tensor from the batch shape cache.
std::unordered_map<std::string, size_t> GetBatchAverageSize(
    const std::vector<std::map<std::string, std::vector<int>>>& batches) {
  std::unordered_map<std::string, size_t> var2size;
  // The average size of the batches for each variable.
  int num_batch = 0;
  for (const auto& batch : batches) {
    num_batch++;
    for (const auto& item : batch) {
      int dim = std::accumulate(item.second.begin(), item.second.end(), 1,
                                [](int a, int b) { return a * b; });
      var2size[item.first] += dim;
    }
  }

  for (auto& item : var2size) {
    item.second /= num_batch;
  }

  return var2size;
}

// Analysis the batch shapes loading from the cache file.
// By splitting the variables to different clusters by analyzing their batch
// size, we can pre-schedule the changes of difference LoDTensor when different
// length of input sequences is entered.
// This should works fine for the models operating on sentences.
std::vector<std::unordered_set<std::string>> AnalysisBatchShapesByBatchSize(
    const std::vector<std::map<std::string, std::vector<int>>>& batches) {
  // collect the batch size of each shape and combine to a stringstream in
  // converient to generate a hash.
  std::unordered_map<std::string, std::stringstream> var_batchsize_hashes;
  for (auto& batch : batches) {
    for (auto& ele : batch) {
      PADDLE_ENFORCE(!ele.second.empty());
      int batch_size = ele.second.front();
      // TODO(Superjomn) might consume large memory here, use combine hash.
      var_batchsize_hashes[ele.first] << batch_size;
    }
  }

  // Split to sets by batch size sequences.
  std::unordered_map<size_t /*hash*/, std::unordered_set<std::string>>
      shape_sets;
  for (auto& ele : var_batchsize_hashes) {
    auto hash = std::hash<std::string>()(ele.second.str());
    shape_sets[hash].insert(ele.first);
  }
  std::vector<std::unordered_set<std::string>> res;
  for (auto& ele : shape_sets) {
    res.emplace_back(std::move(ele.second));
  }

  VLOG(3) << "Cluster by batch_size and get " << res.size() << " clusters";
  return res;
}

// Analysis the batch shapes loading from the cache file, and split them to
// different clusters by their size.
// This should works fine for the overall models.
std::vector<std::unordered_set<std::string>> AnalysisBatchShapesBySimilarSize(
    const space_table_t& space_table,
    const std::vector<std::map<std::string, std::vector<int>>>& batches,
    int interval = 200000) {
  PADDLE_ENFORCE_GT(interval, 0);
  // cluster to different clusters.
  size_t max_size = 0;
  for (auto& item : space_table) {
    max_size = std::max(item.second, max_size);
  }
  VLOG(4) << "tensor max size " << max_size;

  std::vector<std::unordered_set<std::string>> res;

  // cluster by intervals.
  for (size_t interval_size = 0; interval_size <= max_size;
       interval_size += interval) {
    std::unordered_set<std::string> cluster;
    for (auto& item : space_table) {
      if (interval_size <= item.second &&
          interval_size + interval > item.second) {
        cluster.insert(item.first);
      }
    }
    if (!cluster.empty()) {
      res.push_back(cluster);
    }
  }

  VLOG(3) << "Cluster by interval and get " << res.size() << " cluster";
  return res;
}

std::string MemoryOptimizePass::repr() const { return "memory optimize pass"; }

std::pair<size_t, size_t> GetRange(
    const std::unordered_map<std::string, size_t>& ave_size) {
  auto res = std::make_pair(std::numeric_limits<size_t>::max(),
                            std::numeric_limits<size_t>::min());
  for (auto& item : ave_size) {
    res.first = std::min(item.second, res.first);
    res.second = std::max(item.second, res.second);
  }
  return res;
}

void MemoryOptimizePass::RunImpl(Argument* argument) {
  // When force update, should not optimize memory.
  if (!argument->enable_memory_optim() ||
      argument->static_memory_optim_force_update())
    return;
  graph_ = argument->main_graph_ptr();

  auto path = GetMemoryCachePath(
      argument->model_dir_valid() ? argument->model_dir() : "",
      argument->model_program_path_valid() ? argument->model_program_path()
                                           : "");
  VLOG(3) << "Load memory cache from " << path;
  std::vector<std::map<std::string, std::vector<int>>> batches;

  if (!(argument->static_memory_optim() && inference::IsFileExists(path))) {
    string::PrettyLogInfo("--- Performing dynamic memory optimize");
    // batches = FakeBatchVarShapes(argument->main_program());
    int sort_kind = 0;
    std::unordered_map<std::string, lifecycle_t> lifecycles;
    space_table_t space_table;
    std::unordered_map<std::string, std::string> node2cluster;
    std::unordered_map<std::string, int> cluster_size;

    CollectLifeCycle(&lifecycles, sort_kind);
    CollectVarMemorySize(&space_table);
    MakeSimpleReusePlan(lifecycles, space_table, &node2cluster, &cluster_size);
    UpdateOpDescsByReuse(graph_, node2cluster, sort_kind);
    return;

  } else {
    string::PrettyLogInfo("--- Performing static memory optimize");
    batches = DeseralizeBatchVarShapes(path);
  }
  auto var_batch_ave_size = GetBatchAverageSize(batches);

  // Get min and max memory size.
  const auto range = GetRange(var_batch_ave_size);
  const int cluster_size = std::max(
      static_cast<int>((range.second - range.first) / 100 /*cluster num*/),
      1024);
  const int cluster_size1 = std::max(
      static_cast<int>((range.second - range.first) / 1000 /*cluster num*/),
      1024);

  std::unordered_map<std::string, Node*> tensor_nodes;
  space_table_t space_table;
  CollectVarMemorySize(var_batch_ave_size, &tensor_nodes, &space_table);

  std::unordered_map<std::string, std::string> reuse_table;
  double max_saving_ratio = 0.;

  std::vector<std::function<MemoryAllocation()>> strategies;

  for (int sort_kind = 0; sort_kind < 2; sort_kind++) {
    if (argument->static_memory_optim()) {
      // This strategy only make scene in static memory optimize.
      strategies.emplace_back([&, sort_kind] {
        auto clustered_vars_by_batch_size =
            AnalysisBatchShapesByBatchSize(batches);
        MemoryAllocation allocation;
        MakeReusePlan(clustered_vars_by_batch_size, var_batch_ave_size,
                      space_table, &reuse_table, sort_kind, &allocation);
        return allocation;
      });
    }

    strategies.emplace_back([&, sort_kind] {
      auto clustered_vars_by_ave_size =
          AnalysisBatchShapesBySimilarSize(space_table, batches, cluster_size);
      MemoryAllocation allocation;
      MakeReusePlan(clustered_vars_by_ave_size, var_batch_ave_size, space_table,
                    &reuse_table, sort_kind, &allocation);
      return allocation;
    });

    strategies.emplace_back([&, sort_kind] {
      auto clustered_vars_by_ave_size =
          AnalysisBatchShapesBySimilarSize(space_table, batches, cluster_size1);
      MemoryAllocation allocation;
      MakeReusePlan(clustered_vars_by_ave_size, var_batch_ave_size, space_table,
                    &reuse_table, sort_kind, &allocation);
      return allocation;
    });

    strategies.emplace_back([&, sort_kind] {
      auto clustered_vars_by_ave_size = AnalysisBatchShapesBySimilarSize(
          space_table, batches,
          std::numeric_limits<int>::max());  // no intervals
      MemoryAllocation allocation;
      MakeReusePlan(clustered_vars_by_ave_size, var_batch_ave_size, space_table,
                    &reuse_table, sort_kind, &allocation);
      return allocation;
    });
  }

  std::function<MemoryAllocation()>* best_strategy{nullptr};

  // Try all strategies to get the best result.
  for (auto& strategy : strategies) {
    auto allocation = strategy();
    string::PrettyLogDetail("--- get strategy saving %f memory for workspace",
                            allocation.GetSavingRatio());
    if (allocation.GetSavingRatio() > max_saving_ratio) {
      max_saving_ratio = allocation.GetSavingRatio();
      best_strategy = &strategy;
    }
  }
  if (!best_strategy) {
    LOG(ERROR) << "This model makes poor memory optimize, skip memory optimize";
    return;
  }
  auto memory_allocation = (*best_strategy)();

  string::PrettyLogInfo(
      "--- Saved %.2f%s memory for workspace(temporary variables)",
      memory_allocation.GetSavingRatio() * 100, "%");

  argument->main_graph().Set(framework::ir::kGraphToProgramVarsToRemove,
                             new std::unordered_set<std::string>);
  auto& vars2remove =
      argument->main_graph().Get<std::unordered_set<std::string>>(
          framework::ir::kGraphToProgramVarsToRemove);

  PerformReusePlan(reuse_table, memory_allocation.sort_kind, &vars2remove);
  argument->SetMemoryOptimSortKind(memory_allocation.sort_kind);
}

float MemoryOptimizePass::MemoryAllocation::GetSavingRatio() const {
  return (saved / 1024.) / (allocated / 1024. + saved / 1024.);
}
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
