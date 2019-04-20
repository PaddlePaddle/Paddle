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

#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace details {

class ReluMemoryOptimizePass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override;
};

template <bool kIsInput>
static ir::Node *FindReluOutNode(ir::Node *node) {
  auto *op_desc = node->Op();
  std::string var_name =
      kIsInput ? op_desc->Input("Out")[0] : op_desc->Output("Out")[0];
  auto tgt_nodes = kIsInput ? node->inputs : node->outputs;

  ir::Node *found_node = nullptr;
  for (auto *n : tgt_nodes) {
    if (n->Name() == var_name) {
      PADDLE_ENFORCE(found_node == nullptr,
                     "Found duplicate Out(%s) of relu or relu_grad", var_name);
      found_node = n;
    }
  }

  PADDLE_ENFORCE_NOT_NULL(found_node,
                          "Cannot find Out(%s) in relu or relu_grad", var_name);
  return found_node;
}

struct ReluOpInfo {
  ReluOpInfo(ir::Node *relu, ir::Node *relu_grad, ir::Node *out)
      : relu_(relu), relu_grad_(relu_grad), out_(out) {}

  ir::Node *relu_;
  ir::Node *relu_grad_;
  ir::Node *out_;
};

template <typename T>
static void RemoveVectorValue(std::vector<T> *vec, T value) {
  size_t found_num = 0;
  while (true) {
    auto iter = std::find(vec->begin(), vec->end(), value);
    if (iter != vec->end()) {
      vec->erase(iter);
      ++found_num;
    } else {
      PADDLE_ENFORCE(found_num > 0, "Cannot find value in vector");
      break;
    }
  }
}

template <typename T>
static void ReplaceIf(std::vector<T> *vec, T old_value, T new_value) {
  for (auto it = vec->begin(); it != vec->end(); ++it) {
    if (*it == old_value) {
      *it = new_value;
    }
  }
}

static void ReplaceGraphNode(ir::Node *old_node, ir::Node *new_node) {
  for (auto *in_node : old_node->inputs) {
    ReplaceIf(&(in_node->outputs), old_node, new_node);
    new_node->inputs.emplace_back(in_node);
  }

  for (auto *out_node : old_node->outputs) {
    ReplaceIf(&(out_node->inputs), old_node, new_node);
    new_node->outputs.emplace_back(out_node);
  }
}

void ReluMemoryOptimizePass::ApplyImpl(ir::Graph *graph) const {
  const auto &nodes = graph->Nodes();

  // Step 1: find out all relu and relu_grad op
  // relu_ops is a map of VarNode(Out) -> OpNode(relu_op)
  // relu_grad_ops is a map of VarNode(Out) -> OpNode(relu_grad_op)
  std::unordered_map<ir::Node *, ir::Node *> relu_ops, relu_grad_ops;

  for (auto *node : nodes) {
    if (!node->IsOp() || !node->Op()) {
      continue;
    }

    auto op_type = node->Op()->Type();
    if (op_type == "relu") {
      ir::Node *relu_out = FindReluOutNode<false>(node);
      PADDLE_ENFORCE(relu_ops.count(relu_out) == 0,
                     "Found duplicate relu op outputs the same ir::Node, the "
                     "graph may be wrong");
      relu_ops.insert({relu_out, node});
    } else if (op_type == "relu_grad") {
      ir::Node *relu_out = FindReluOutNode<true>(node);
      PADDLE_ENFORCE(relu_grad_ops.count(relu_out) == 0,
                     "Found duplicate relu_grad op with the same input Out, "
                     "the graph may be wrong.");
      relu_grad_ops.insert({relu_out, node});
    }
  }

  // Step 2: Find the matched relu_grad and relu according to Out node
  std::vector<ReluOpInfo> relu_infos;

  for (auto &relu_grad_op : relu_grad_ops) {
    auto *relu_out = relu_grad_op.first;
    auto iter = relu_ops.find(relu_out);
    PADDLE_ENFORCE(iter != relu_ops.end(),
                   "Cannot find matched relu op for relu_grad op");
    relu_infos.emplace_back(iter->second, relu_grad_op.second, relu_out);
    relu_ops.erase(relu_out);
  }

  // Step 3: Replace relu with relu2, relu_grad with relu2_grad
  size_t relu_mask_id = 0;
  for (auto &info : relu_infos) {
    auto *relu = info.relu_;
    auto *relu_grad = info.relu_grad_;
    auto *out = info.out_;

    // Create new OpDesc
    OpDesc relu_desc(*(relu->Op()));
    relu_desc.SetType("relu2");

    OpDesc relu_grad_desc(relu_grad->Op()->Block());
    relu_grad_desc.SetType("relu2_grad");
    relu_grad_desc.SetInput(GradVarName("Out"),
                            relu_grad->Op()->Input(GradVarName("Out")));
    relu_grad_desc.SetOutput(GradVarName("X"),
                             relu_grad->Op()->Output(GradVarName("X")));
    relu_grad_desc.SetAttrMap(relu_grad->Op()->GetAttrMap());

    auto *block = relu_desc.Block();
    PADDLE_ENFORCE(block == relu_grad_desc.Block(),
                   "relu and relu_grad must locate in the same block");
    std::string mask_var_name = "@RELU_MASK@" + std::to_string(relu_mask_id++);
    PADDLE_ENFORCE(!block->HasVarRecursive(mask_var_name),
                   "The program contains unsupported variable name %s. Maybe "
                   "you have applied relu_memory_optimize_pass twice!",
                   mask_var_name);

    // Create a VarDesc to represent the new mask var
    // FIXME(zjl): create a variable in block to fix memory_optimize_pass
    // cannot find variable in block error.
    VarDesc *mask = block->Var(mask_var_name);
    mask->SetShape({0});
    mask->SetType(proto::VarType::LOD_TENSOR);
    mask->SetPersistable(false);
    mask->SetDataType(proto::VarType::UINT8);

    // Change OpDesc in/out
    relu_desc.SetOutput("Mask", {mask->Name()});
    relu_desc.Flush();

    relu_grad_desc.SetInput("Mask", {mask->Name()});
    relu_grad_desc.Flush();

    // Create a new op node to the graph
    auto *new_relu = graph->CreateOpNode(&relu_desc);
    auto *new_relu_grad = graph->CreateOpNode(&relu_grad_desc);
    ReplaceGraphNode(relu, new_relu);
    ReplaceGraphNode(relu_grad, new_relu_grad);

    // Add a new var node to the graph
    ir::Node *mask_node = graph->CreateVarNode(mask);
    new_relu->outputs.emplace_back(mask_node);
    new_relu_grad->inputs.emplace_back(mask_node);
    mask_node->inputs.emplace_back(new_relu);
    mask_node->outputs.emplace_back(new_relu_grad);

    // remove out dependency of relu_grad
    RemoveVectorValue(&(new_relu_grad->inputs), out);
    RemoveVectorValue(&(out->outputs), new_relu_grad);

    graph->RemoveNode(relu);
    graph->RemoveNode(relu_grad);
    VLOG(2) << "Replace relu with relu2 in op with output " << out->Name();
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(relu_memory_optimize_pass,
              paddle::framework::details::ReluMemoryOptimizePass);
