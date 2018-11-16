#include "paddle/fluid/inference/analysis/ir_passes/mem_optim_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_traits.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Graph;
using framework::ir::Node;

const int kBatchSize = 13;  // replace the placement -1 in shape

std::unique_ptr<framework::ir::Graph> MemOptimPass::ApplyImpl(
    std::unique_ptr<framework::ir::Graph> graph) const {
  return Pass::ApplyImpl(graph);
}

// Traverse the graph in topological order.
void MemOptimPass::CollectLifeCycle(
    std::unordered_map<std::string, lifecycle_t>* lifecycles) {
  max_lifecycle_ = 0;
  for (auto* op_node : framework::ir::TopologySortOperations(*graph_)) {
    auto reads = op_node->inputs;
    auto writes = op_node->outputs;

    std::vector<std::string> requires(reads.begin(), reads.end());
    requires->insert(std::back_inserter(requires), writes.begin(),
                     writes.end());

    for (const std::string& var : requires) {
      if (!lifecycles->count(var)) {
        (*lifecycles)[var] = std::make_pair(max_lifecycle_, -1);
      } else {
        (*lifecycles)[var].second = max_lifecycle_;  // max()
      }
    }

    ++max_lifecycle_;
  }
}

void MemOptimPass::CollectShapes(
    std::unordered_map<std::string, Node*>* tensor_nodes) {
  // Collect tensors from graph.
  for (auto* node : graph_->Nodes()) {
    if (node->IsVar() &&
        node->Var()->GetType() ==
            framework::proto::VarType::Type::VarType_Type_LOD_TENSOR) {
      (*tensor_nodes)[node->Name()] = node;
    }
  }
}

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
      return -1;
  }
}

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

__attribute__((warn_unused_result)) bool FindSutableTensorToReuse(
    int space_required,
    const std::unordered_map<std::string, Node*>& tensor_nodes,
    std::unordered_set<std::string>* free_existing_tensors,
    std::unordered_map<std::string, int> space_table,
    std::string* tensor_to_reuse) {
  std::pair<std::string, int> best_fit;
  best_fit.second = std::numeric_limits<int>::max();

  for (auto& tensor : *free_existing_tensors) {
    int space = space_table[tensor];
    int space_diff = space_required - space;
    if (space_diff > 0 && space_diff < best_fit.second) {
      best_fit.first = tensor;
      best_fit.second = space_diff;
    }
  }

  if (best_fit.second < std::numeric_limits<int>::max()) {
    *tensor_to_reuse = best_fit.first;
    return true;
  }
  return false;
}

void AllocateNewTensor(
    const std::string& name, int space_required,
    const std::unordered_map<std::string, Node*>& tensor_nodes,
    std::unordered_set<std::string>* free_existing_tensors,
    std::unordered_map<std::string, int>* space_table
    ) {
  free_existing_tensors->insert(name);
  space_table->emplace(name, space_required);
}

void MemOptimPass::MakeReusePlan() {
  std::unordered_map<std::string, lifecycle_t> lifecycles;
  std::unordered_map<std::string, Node*> tensor_nodes;
  // The allocated tensors whose memory can be reused, they will live across the
  // program execution.
  std::unordered_set<std::string> existing_tensors;
  // The existing tensor that has been allocated, and is also free to reuse.
  std::unordered_set<std::string> free_existing_tensors;
  // var_name -> reused_var_name
  std::unordered_map<std::string, std::string> reuse_table;
  std::unordered_map<std::string, int> space_table;

  CollectLifeCycle(&lifecycles);
  CollectShapes(&tensor_nodes);

  for (int age = 0; age < max_lifecycle_; ++age) {
    for (auto elem_it = lifecycles.begin(); elem_it != lifecycles.end();
         elem_it++) {
      // Collect dead tensors.
      if (elem_it->second.second < age) {
        curr_dead_tensors.insert(elem_it->first);
        elem_it = lifecycles.erase(elem_it);
      }

      // Collect born tensors
      std::unordered_set<std::string> born_tensors;
      if (elem_it->second.first == age) {
        born_tensors.insert(elem_it->first);
      }

      // Reuse the dead tensors for born_tensors
    }
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
