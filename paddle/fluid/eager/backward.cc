// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/backward.h"
#include <queue>

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/grad_tensor_holder.h"
#include "paddle/fluid/eager/utils.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"

namespace egr {

std::unordered_map<GradNodeBase*, int> getInDegreeMap(
    const std::queue<GradNodeBase*>& init_queue) {
  // Calculate in_degree for each node
  // We can completely remove this pass, if in_degree were set during forward
  // pass
  std::unordered_map<GradNodeBase*, int> node_in_degree_map;

  // Copy nodes
  std::queue<GradNodeBase*> queue = init_queue;
  std::unordered_set<GradNodeBase*> visited;

  // Visit each node exactly once in any order
  while (!queue.empty()) {
    GradNodeBase* node = queue.front();
    queue.pop();

    if (visited.count(node)) {
      continue;
    }
    visited.insert(node);

    // Find and append next nodes
    const std::vector<std::vector<Edge>>& edges = node->GetEdges();
    for (const auto& edge_list : edges) {
      for (const Edge& edge : edge_list) {
        GradNodeBase* next_node = edge.GetMutableGradNode().get();

        // Next node could be nullptr if it is leaf tensor with no
        // AccumulationNode attached
        // Or it could also originated from dispensable inputs
        if (!next_node) continue;

        // Update in_degree
        if (!node_in_degree_map.count(next_node))
          node_in_degree_map[next_node] = 0;
        node_in_degree_map[next_node]++;
        queue.push(next_node);
      }
    }
  }

  return node_in_degree_map;
}

void RunBackwardHooks(
    const std::vector<std::vector<egr::EagerTensor>>& grad_tensors,
    egr::GradNodeBase* grad_node) {
  grad_node->ApplyGradientHooks(grad_tensors);
  VLOG(6) << "Apply Reduce Hooks for node";
  grad_node->ApplyReduceHooks();
}

void RunBackward(const std::vector<egr::EagerTensor>& tensors,
                 const std::vector<egr::EagerTensor>& grad_tensors,
                 bool retain_graph) {
  VLOG(6) << "Start Backward";
  // *Gradient Hook should happen at node-level
  // *Inplace version check should perform at node-level
  // *Cross-batch accumulation happens at forward pass

  /* --- Initialization --- */
  // 1. Init queue with starting nodes
  // 2. Prepare initial input buffers
  std::queue<GradNodeBase*> queue;
  std::unordered_map<GradNodeBase*, std::unique_ptr<GradTensorHolder>>
      node_input_buffers_dict;
  for (size_t i = 0; i < tensors.size(); i++) {
    const egr::EagerTensor& tensor = tensors[i];

    AutogradMeta* auto_grad_meta = EagerUtils::unsafe_autograd_meta(tensor);
    // Get grad input info from target tensors
    auto input_info = auto_grad_meta->OutRankInfo();

    VLOG(2) << "Out Rank of Tensor is slot: " << input_info.first
            << ", rank: " << input_info.second;
    // Get target GradNodeBase from target tensors
    GradNodeBase* grad_node = auto_grad_meta->GetMutableGradNode().get();

    // Prepare GradTensorHolder
    if (!node_input_buffers_dict.count(grad_node)) {
      VLOG(6) << "Create Value for grad input tensor " << i;
      node_input_buffers_dict[grad_node] =
          std::make_unique<GradTensorHolder>(grad_node->InputMeta());
    }

    if (grad_tensors.size() > 0) {
      PADDLE_ENFORCE(
          grad_tensors.size() == tensors.size(),
          paddle::platform::errors::Fatal(
              "Detected size mismatch between tensors and grad_tensors"
              "grad_tensors should either have "
              "size = 0 or same size as tensors"));
      // Feed given tensor if it's provided
      VLOG(6) << "Fill grad input tensor " << i << "with give grad tensor";
      node_input_buffers_dict[grad_node]->add(
          input_info.first, input_info.second, grad_tensors[i]);

    } else {
      VLOG(6) << "Fill grad input tensor " << i << " with 1.0";
      // Initialize tensor with 1.0
      // Forward Tensor "tensor" is passed to indicate tensortype, datatype and
      // dims
      // GradTensorHolder will initialize another tensor with same tensortype,
      // datatype and dims but filled with 1.0
      node_input_buffers_dict[grad_node]->add(
          input_info.first, input_info.second, tensor, true /*fill_one=true*/);
    }

    // Prepare queue
    queue.push(grad_node);
  }

  VLOG(6) << "Update In degree Map for backward";
  // 3. Compute in_degree for each node
  std::unordered_map<GradNodeBase*, int> node_in_degree_map =
      getInDegreeMap(queue);

  /* --- Topological Visit --- */
  // 1. Pop queue
  // 2. Run node
  //    |- node(grads)
  //    |- Prepare for next node
  // 3. Update queue
  VLOG(6) << "Run Backward";
  while (!queue.empty()) {
    GradNodeBase* node = queue.front();
    queue.pop();

    // Run node: This is where Hook happens
    PADDLE_ENFORCE(
        node_input_buffers_dict.count(node),
        paddle::platform::errors::Fatal(
            "Unable to find next node in the InputBuufer"
            "Trying to run Node without configuring its GradTensorHolder"));

    std::unique_ptr<GradTensorHolder> node_input_buffer =
        std::move(node_input_buffers_dict[node]);
    VLOG(6) << "Run Backward Kernel with input_buffer";

    RunBackwardHooks(node_input_buffer->Buffers(), node);
    // TODO(jiabin): Support post hook here and make hook run in seperate
    // operator
    // Run Pre Backward Node and get outputs
    std::vector<std::vector<egr::EagerTensor>> grad_output_tensors =
        (*node)(node_input_buffer->Buffers());
    // TODO(jiabin): Should we erase it or find a more efficient way.
    node_input_buffers_dict.erase(node);

    // Prepare GradTensorHolder for next node
    const std::vector<std::vector<Edge>>& edges = node->GetEdges();

    PADDLE_ENFORCE(edges.size() == grad_output_tensors.size() || edges.empty(),
                   paddle::platform::errors::Fatal(
                       "Number of edges should be either empty ( for leaf node "
                       ") or the same as number of output grad tensors"));

    for (size_t i = 0; i < edges.size(); i++) {
      for (size_t j = 0; j < edges[i].size(); j++) {
        const Edge& edge = edges[i][j];
        auto edge_rank = edge.GetEdgeRankInfo();
        // Since we make edge has as same rank as bwd outputs, we indexing them
        // with
        // the same rank(i, j)
        VLOG(6) << "Get Edge with slot: " << i << ", rank: " << j;
        egr::EagerTensor& grad_output_tensor = grad_output_tensors[i][j];
        if (!grad_output_tensor.defined() ||
            !grad_output_tensor.initialized()) {
          VLOG(6) << "We get grad_output_tensor with slot: " << i
                  << ", rank: " << j << " as uninitialized or undefined tensor";
        }
        GradNodeBase* next_node = edge.GetMutableGradNode().get();

        // Next node could be nullptr if it is leaf tensor with no
        // AccumulationNode attached
        // Or it could also originated from dispensable inputs
        if (!next_node) continue;

        if (!node_input_buffers_dict.count(next_node)) {
          node_input_buffers_dict[next_node] =
              std::make_unique<GradTensorHolder>(next_node->InputMeta());
        }
        VLOG(6) << "Sum grad inputs for edge slot: " << edge_rank.first
                << ", rank: " << edge_rank.second;
        node_input_buffers_dict[next_node]->add(
            edge_rank.first, edge_rank.second, grad_output_tensor);

        // Update queue
        node_in_degree_map[next_node]--;
        PADDLE_ENFORCE(node_in_degree_map[next_node] >= 0,
                       paddle::platform::errors::Fatal(
                           "Detected in-degree value smaller than zero."
                           "Node's in-degree cannot be negative"));
        if (node_in_degree_map[next_node] == 0) {
          queue.emplace(std::move(next_node));
        }
      }
    }
  }
}

}  // namespace egr
