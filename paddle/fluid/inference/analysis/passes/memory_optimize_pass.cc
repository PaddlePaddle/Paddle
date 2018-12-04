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
#include <fstream>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Graph;
using framework::ir::Node;

const int kBatchSize = 13;  // replace the placement -1 in shape

// Several kinds of topological sort.
std::vector<Node*> TopoSort(const Graph& graph, int sort_kind) {
  switch (sort_kind) {
    case 0:
      return framework::ir::TopologySortOperations(graph);
    default:
      return framework::ir::TopologyDfsSortOperations(graph);
  }
}

// Collect the lifecycles of the tensors.
// Traverse the graph in topological order.
// TODO(Superjomn) the traversal order also affect the lifecycles, try to change
// the order to improve the reuse performance latter.
void MemoryOptimizePass::CollectLifeCycle(
    std::unordered_map<std::string, lifecycle_t>* lifecycles,
    int sort_kind) const {
  max_lifecycle_ = 0;
  for (auto* op_node : TopoSort(*graph_, sort_kind)) {
    if (!op_node->IsOp()) continue;
    auto reads = op_node->inputs;
    auto writes = op_node->outputs;

    std::vector<Node*> requires(reads.begin(), reads.end());
    requires.insert(requires.end(), writes.begin(), writes.end());

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
          int dead = max_lifecycle_;
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
int ShapeToSpace(const std::vector<int64_t>& shape,
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
void MemoryOptimizePass::CollectShapes(
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
bool FindSuitableTensorToReuse(
    const std::string& tensor, int space_required,
    const std::unordered_map<std::string, Node*>& tensor_nodes,
    std::unordered_set<std::string>* free_existing_tensors,
    const std::unordered_map<std::string, int>& space_table,
    const std::vector<std::unordered_set<std::string>>& var_clusters,
    std::string* tensor2use) __SHOULD_USE_RESULT__;

bool FindSuitableTensorToReuse(
    const std::string& tensor, int space_required,
    const std::unordered_map<std::string, Node*>& tensor_nodes,
    std::unordered_set<std::string>* free_existing_tensors,
    const std::unordered_map<std::string, int>& space_table,
    const std::vector<std::unordered_set<std::string>>& var_clusters,
    std::string* tensor2use) {
  std::pair<std::string, int> best_fit;
  best_fit.second = std::numeric_limits<int>::max();
  VLOG(3) << "Split Tensors to " << var_clusters.size() << " clusters";

  // find the cluster this var belongs to.
  const std::unordered_set<std::string>* cluster = nullptr;
  for (const auto& c : var_clusters) {
    if (c.count(tensor)) {
      cluster = &c;
      break;
    }
  }
  PADDLE_ENFORCE_NOT_NULL(
      cluster,
      "something wrong in memory optimization, the variable cluster phase.");

  for (auto& candidate : *free_existing_tensors) {
    if (!space_table.count(candidate)) continue;
    int space = space_table.at(candidate);
    int space_diff = space - space_required;
    if (space_diff >= 0 && space_diff < best_fit.second &&
        cluster->count(candidate)) {
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

// Calculate the memory usage.
void MemoryEvaluate(
    const std::unordered_map<std::string, std::string>& reuse_table,
    const std::unordered_map<std::string, int>& space_table,
    const std::unordered_map<std::string, size_t>& var_batch_ave_size,
    size_t* allocated, size_t* saved) {
  *allocated = 0;
  *saved = 0;

  for (auto elem : reuse_table) {
    // if (elem.first == "feed" || elem.second == "second") continue;
    if (elem.first == elem.second) {
      *allocated += var_batch_ave_size.at(elem.first);
    } else {
      *saved += var_batch_ave_size.at(elem.first);
      VLOG(4) << "reuse " << elem.first << " -> " << elem.second;
    }
  }
}

// Return saved ratio.
void MemoryOptimizePass::MakeReusePlan(
    const std::vector<std::unordered_set<std::string>>& var_clusters,
    const std::unordered_map<std::string, size_t>& var_batch_ave_size,
    std::unordered_map<std::string, std::string>* reuse_table, int sort_kind,
    MemoryAllocation* memory_allocation) const {
  // Clear the existing plan.
  reuse_table->clear();

  std::unordered_map<std::string, lifecycle_t> life_cycles;
  std::unordered_map<std::string, Node*> tensor_nodes;
  // The allocated tensors whose memory can be reused, they will live across the
  // program execution.
  std::unordered_set<std::string> existing_tensors;
  // The existing tensor that has been allocated, and is also free to reuse.
  std::unordered_set<std::string> free_existing_tensors;
  std::unordered_map<std::string, int> space_table;

  CollectLifeCycle(&life_cycles, sort_kind);
  CollectShapes(&tensor_nodes, &space_table);

  for (int age = 0; age < max_lifecycle_; ++age) {
    std::unordered_set<std::string> born_tensors;
    std::unordered_set<std::string> dead_tensors;
    // Gather the dead and born tensors.
    for (auto elem_it = life_cycles.begin(); elem_it != life_cycles.end();
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
      if (FindSuitableTensorToReuse(tensor, space_required, tensor_nodes,
                                    &free_existing_tensors, space_table,
                                    var_clusters, &tensor2reuse)) {
        if (tensor != tensor2reuse) {
          VLOG(4) << tensor << " -> " << tensor2reuse;
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

  size_t allocated, saved_memory;
  MemoryEvaluate(*reuse_table, space_table, var_batch_ave_size, &allocated,
                 &saved_memory);
  float saved_ratio =
      static_cast<float>(saved_memory) / (saved_memory + allocated);
  memory_allocation->allocated = allocated / 1024. / 1024.;
  memory_allocation->saved = saved_memory / 1024. / 1024.;
  memory_allocation->saving_ratio = saved_ratio;
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
  for (auto* node : TopoSort(*graph, sort_kind)) {
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

void MemoryOptimizePass::PerformReusePlan(
    const std::unordered_map<std::string, std::string>& reuse_table,
    int sort_kind) const {
  std::unordered_map<std::string, Node*> var_node_table;
  BuildVarNodeTable(graph_, &var_node_table);
  // UpdateIrGraphByReuse(graph_, reuse_table, var_node_table);
  UpdateOpDescsByReuse(graph_, reuse_table, sort_kind);
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
    for (auto var_info : split(line, ';')) {
      auto fields = split(var_info, ':');
      PADDLE_ENFORCE_EQ(fields.size(), 2UL);
      auto var_name = fields.front();
      auto shape_str = split(fields[1], ',');
      std::vector<int> shape;
      for (auto v : shape_str) shape.push_back(std::stoi(v));
      batch[var_name] = shape;
    }
    batch_shapes.push_back(batch);
  }
  return batch_shapes;
}

// Calculate the average memory size of each variable from the batch shape
// cache.
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
      var2size[item.first] =
          var2size[item.first] * (num_batch - 1) / num_batch + dim / num_batch;
    }
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
  return res;
}

// Analysis the batch shapes loading from the cache file, and split them to
// different clusters by their size.
// This should works fine for the overall models.
std::vector<std::unordered_set<std::string>> AnalysisBatchShapesBySimilarSize(
    const std::vector<std::map<std::string, std::vector<int>>>& batches,
    int interval = 2000 /*1kb*/) {
  // cluster to different clusters.
  size_t max_size = 0;
  auto var2size = GetBatchAverageSize(batches);
  for (auto& item : var2size) {
    max_size = std::max(item.second, max_size);
  }

  std::vector<std::unordered_set<std::string>> res;

  // cluster by intervals.
  for (size_t interval_size = 0; interval_size < max_size;
       interval_size += interval) {
    std::unordered_set<std::string> cluster;
    for (auto& item : var2size) {
      if (interval_size <= item.second &&
          interval_size + interval > item.second) {
        cluster.insert(item.first);
      }
    }
    if (!cluster.empty()) {
      res.push_back(cluster);
    }
  }

  LOG(INFO) << "Cluster by interval and get " << res.size() << " cluster";
  return res;
}

std::string MemoryOptimizePass::repr() const { return "memory optimize pass"; }

void MemoryOptimizePass::RunImpl(Argument* argument) {
  if (!argument->enable_memory_optim()) {
    return;
  }
  auto path = GetMemoryCachePath(
      argument->model_dir_valid() ? argument->model_dir() : "",
      argument->model_program_path_valid() ? argument->model_program_path()
                                           : "");
  VLOG(3) << "Load memory cache from " << path;
  if (inference::IsFileExists(path)) {
    VLOG(4) << "Performing memory optimize";
    auto batches = DeseralizeBatchVarShapes(path);
    auto clustered_vars_by_batch_size = AnalysisBatchShapesByBatchSize(batches);
    auto clustered_vars_by_ave_size = AnalysisBatchShapesBySimilarSize(batches);
    auto var_batch_ave_size = GetBatchAverageSize(batches);

    graph_ = argument->main_graph_ptr();
    std::unordered_map<std::string, std::string> reuse_table;
    float max_saving_ratio = 0.;

    std::vector<std::function<MemoryAllocation()>> strategies;

    for (int sort_kind = 0; sort_kind < 2; sort_kind++) {
      strategies.emplace_back([&, sort_kind] {
        MemoryAllocation allocation;
        MakeReusePlan(clustered_vars_by_batch_size, var_batch_ave_size,
                      &reuse_table, sort_kind, &allocation);
        return allocation;
      });

      strategies.emplace_back([&, sort_kind] {
        MemoryAllocation allocation;
        MakeReusePlan(clustered_vars_by_ave_size, var_batch_ave_size,
                      &reuse_table, sort_kind, &allocation);
        return allocation;
      });
    }

    std::function<MemoryAllocation()>* best_strategy;

    // Try all strategies to get the best result.
    for (auto& strategy : strategies) {
      auto allocation = strategy();
      LOG(INFO) << "get strategy saving " << allocation.saving_ratio;
      if (allocation.saving_ratio > max_saving_ratio) {
        max_saving_ratio = allocation.saving_ratio;
        best_strategy = &strategy;
      }
    }

    auto memory_allocation = (*best_strategy)();

    string::PrettyLogDetail(
        "---  Saved %f memory for workspace(temporary variables)",
        memory_allocation.saving_ratio);

    PerformReusePlan(reuse_table, 0);
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
