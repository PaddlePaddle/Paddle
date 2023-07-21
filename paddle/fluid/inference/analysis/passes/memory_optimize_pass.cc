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

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/inference/analysis/pass_result_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/inference/utils/io_utils.h"


namespace paddle {
namespace framework {
namespace ir {
class Graph;
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

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
    Argument *argument,
    std::unordered_map<std::string, lifecycle_t>* lifecycles,
    int sort_kind) const {
  framework::ir::Graph* graph = argument->main_graph_ptr();
  int max_lifecycle = 0;
  double persis_byte = 0;
  for (auto* op_node : framework::ir::TopologyVarientSort(
           *graph, static_cast<framework::ir::SortKind>(sort_kind))) {
    // std::cout << "tuopu: " << op_node->Name() << std::endl;
    if (!op_node->IsOp()) continue;
    auto reads = op_node->inputs;
    auto writes = op_node->outputs;

    std::vector<Node*>
    requires(reads.begin(), reads.end());
    requires.insert(requires.end(), writes.begin(), writes.end());

    // Disable reuse of feed variables.
    if (op_node->Name() == "feed") {
      for (auto* node : op_node->outputs) {
        auto var = node->Name();
        lifecycles->emplace(var,
                            std::make_pair(0, std::numeric_limits<int>::max()));
        // std::cout << "tuopu: " << var << std::endl;
      }
    } else {
      // Normal operators.
      for (const Node* node : requires) {
        if (!node->Var()) continue;
        if (node->Var()->Persistable()) {
          // "Getting 'tensor_desc' is not supported by the fetch type
          // variable."
          bool is_break = false;
          for (auto op_op : node->inputs) {
            if (op_op->Name() == "fetch") is_break = true;
          }
          if (is_break) continue;

          auto in_shape = node->Var()->GetShape();
          for (auto i : in_shape) {
            CHECK_GT(i, 0);
          }
          auto var_bytes = std::accumulate(in_shape.begin(),
                                           in_shape.end(),
                                           (int64_t)1,
                                           std::multiplies<int64_t>());
          persis_byte +=
              paddle::framework::SizeOfType(node->Var()->GetDataType()) *
              var_bytes;
          continue;
        }
        std::string var = node->Name();
        // std::cout << "tuopu: " << var << std::endl;
        if (!lifecycles->count(var)) {
          (*lifecycles)[var] = std::make_pair(max_lifecycle, max_lifecycle);
        } else {
          (*lifecycles)[var].second =
              std::max(max_lifecycle, lifecycles->at(var).second);  // max()
        }
      }
    }

    ++max_lifecycle;
  }
  LOG(INFO) << "The persistable params in main graph are : "
            << (persis_byte / (1 << 20)) << "MB";
}

void MemoryOptimizePass::CollectVarInfo(
                                                Argument *argument, 
                                                space_table_t* space_table, 
                                                shape_table_t* shape_table, 
                                                dtype_table_t *dtype_info) const {
  framework::ir::Graph* graph = argument->main_graph_ptr();
  const int fake_batch_size = 1;
  
  std::map<std::string, std::vector<int32_t>> opt_shape;

  if(argument->tensorrt_tuned_dynamic_shape()){ // turn on tensorrt dynamic shape
    // get shape information and dtype information from the specified file.
    DeserializeShapeRangeInfo(argument->tensorrt_shape_range_info_path(), 
                                    nullptr, 
                                    &opt_shape, 
                                    nullptr, 
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    dtype_info);
  }

  auto valid_var = [&](framework::ir::Node* node) -> bool {
    // lod operator reuse may cause unknown errors.
    std::set<std::string> invalid_op = {"while",
                                        "conditional_block",
                                        "tensorrt_engine",
                                        "conditional_block_infer",
                                        "merge_lod_tensor_infer",
                                        "merge_lod_tensor",
                                        "equal",
                                        "sequence_pool",
                                        "recurrent",
                                        "lod_reset",
                                        "fetch",
                                        "share_data"};
    for (auto* tmp : node->inputs) {
      CHECK(tmp->IsOp());
      std::string op_type = tmp->Op()->Type();
      if (std::find(invalid_op.begin(), invalid_op.end(), op_type) !=
          invalid_op.end()) {
        return false;
      }
    }
    for (auto* tmp : node->outputs) {
      CHECK(tmp->IsOp());
      std::string op_type = tmp->Op()->Type();
      if (std::find(invalid_op.begin(), invalid_op.end(), op_type) !=
          invalid_op.end()) {
        return false;
      }
    }
    return true;
  };

  // MemoryOptimizePass surppose input model is directed acyclic graph
  // although it's not always the case. so black list is the best compromise
  // between performance and underlying principle.
  std::unordered_set<std::string> black_list;
  for (auto* node : graph->Nodes()) {
    if (node->IsVar() && node->Var() &&
        node->Var()->GetType() ==
            framework::proto::VarType::Type::VarType_Type_LOD_TENSOR) {
      if (!valid_var(node)) {
        black_list.emplace(node->Var()->Name());
      }
    }
  }

  // Collect tensors from graph.
  for (auto* node : graph->Nodes()) {
    if (node->IsVar() && node->Var() &&
        node->Var()->GetType() ==
            framework::proto::VarType::Type::VarType_Type_LOD_TENSOR &&
        !black_list.count(node->Var()->Name())) {
      // Parameters will not be reused.
      if (node->Var()->Persistable()) continue;

      std::vector<int32_t> shape;
      if(argument->tensorrt_tuned_dynamic_shape() && 
          opt_shape.count(node->Var()->Name())){ // turn on tensorrt dynamic shape
        shape = opt_shape[node->Var()->Name()];
      } else{ //turn off tensorrt dynamic shape
        // int64 -> int32
        std::vector<int64_t> shape_int64 = node->Var()->GetShape();
        std::transform(shape_int64.begin(), shape_int64.end(), back_inserter(shape), [](int64_t i){
          return static_cast<int32_t>(i);
        }); 

        for (auto& v : shape) {
          if (v < 0) v = fake_batch_size; // If TRT dynamic shape is not enabled, 
                                          // then set the dimensions with a value of -1 in the varaible to 1
        }
        
      } 

      (*shape_table)[node->Var()->Name()] = shape;
      
      int size = std::accumulate(
          shape.begin(), shape.end(), 1, std::multiplies<int>());
      (*space_table)[node->Var()->Name()] =
          size * paddle::framework::SizeOfType(node->Var()->GetDataType());
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
    if (!space_table.count(data.first)) continue;
    MemNode temp_node;
    temp_node.name = data.first;
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
    // std::cout << "-----" << mem_nodes[i].name << std::endl;
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

std::string MemoryOptimizePass::repr() const { return "memory_optimize_pass"; }

void MemoryOptimizePass::RunImpl(Argument* argument) {
  // Memory optimization.
  // We will perform the following operation:
  // 1. Collect all var's lifetime.
  // 2. Make reuse plan: the vars can be reused if there is no overlap(on
  // lifetime) between
  // them.
  // The final plan is a mapping table in which the key represents the original
  // name of var and the value in the table represents the current name of var.
  // 3. Perform reuse plan: Replace all var's name in the model according to the
  // mapping table.
  if (!argument->enable_memory_optim()) return;
  // Because of pass is a singleton, graph can not be member
  // variables, otherwise, errors will be caused under multithreading
  // conditions.

  int sort_kind = 0;
  std::unordered_map<std::string, lifecycle_t> lifecycles;
  space_table_t space_table;
  shape_table_t shape_table;
  dtype_table_t dtype_table;
  std::unordered_map<std::string, std::string> node2cluster;
  std::unordered_map<std::string, int> cluster_size;

  CollectLifeCycle(argument, &lifecycles, sort_kind);
  CollectVarInfo(argument, &space_table, &shape_table, &dtype_table);
  MakeSimpleReusePlan(lifecycles, space_table, &node2cluster, &cluster_size);

  auto* pass_res_info = PassResultInfoForRuntime::Instance();
  pass_res_info->Set(
      argument->root_predictor_id(), "memory_optimize_pass_node2cluster", node2cluster);
  pass_res_info->Set(
      argument->root_predictor_id(), "memory_optimize_pass_shape_table", shape_table);
      pass_res_info->Set(
      argument->root_predictor_id(), "memory_optimize_pass_dtype_table", dtype_table);

  return;
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
