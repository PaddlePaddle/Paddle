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
  size_t potential_startup_ops_cnt = queue.size();
  size_t cnt = 0;

  // Visit each node exactly once in any order
  while (!queue.empty()) {
    GradNodeBase* node = queue.front();
    queue.pop();

    if (cnt < potential_startup_ops_cnt) {
      node_in_degree_map[node] = 0;
      cnt++;
    }

    if (visited.count(node)) {
      continue;
    }
    visited.insert(node);

    PADDLE_ENFORCE_NOT_NULL(
        node,
        paddle::platform::errors::Fatal(
            "We got null node when we traverse the backward graph, and this "
            "should not happened please check your code and contact us."));
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

// Remove some nodes those doesn't need to be
// stored in potential_stop_nodes、potential_startup_nodes
void UpdateGraphInfo(
    std::unordered_map<GradNodeBase*, AutogradMeta*>*
        target_nodes_inputmeta_map,
    std::unordered_map<GradNodeBase*, std::unordered_set<GradNodeBase*>>*
        depending_nodes,
    std::unordered_set<GradNodeBase*>* potential_stop_nodes,
    std::unordered_set<GradNodeBase*>* potential_startup_nodes) {
  // Updated potential_sotp_nodes by depending_nodes,
  // make sure the path from root to target_node is ok
  std::unordered_set<GradNodeBase*> _startup_ops;
  VLOG(6) << "Running in UpdateGraphInfo";
  std::queue<GradNodeBase*> queue;
  for (auto& target_nodes_inputmeta_pair : *target_nodes_inputmeta_map) {
    queue.emplace(target_nodes_inputmeta_pair.first);
  }

  while (!queue.empty()) {
    auto* target_node = queue.front();
    queue.pop();
    if (!(*depending_nodes)[target_node].empty()) {
      auto precedding_nodes = (*depending_nodes)[target_node];
      for (auto pre_nodes : precedding_nodes) {
        queue.emplace(pre_nodes);
        if (potential_stop_nodes->find(pre_nodes) !=
            potential_stop_nodes->end()) {
          potential_stop_nodes->erase(pre_nodes);
        }
      }
    } else {  // startup_ops have no precedding nodes
      VLOG(6) << "Emplace _startup_ops";
      _startup_ops.emplace(target_node);
    }
  }
  // Purify potential_startup_nodes again, remove some
  // potential startup_nodes that unreach to input target nodes
  if (!_startup_ops.empty()) {
    std::unordered_set<GradNodeBase*> potential_startup_nodes_to_be_erased;
    for (auto node : *potential_startup_nodes) {
      if (_startup_ops.count(node) == 0) {
        VLOG(6) << "Set up potential_startup_nodes_to_be_erased";
        potential_startup_nodes_to_be_erased.emplace(node);
      }
    }
    if (!potential_startup_nodes_to_be_erased.empty()) {
      for (auto node : potential_startup_nodes_to_be_erased) {
        VLOG(6) << "Erase nodes in potential_startup_nodes_to_be_erased";
        potential_startup_nodes->erase(node);
      }
    }
  }
}

// Get Graph Info Betweent input target gradnode and outputs，
// record depending_nodes、 potential_stop_nodes、potential_startup_nodes
void GetGraphInfoBetweenTargets(
    const std::queue<GradNodeBase*>& init_queue,
    std::unordered_map<GradNodeBase*, AutogradMeta*>*
        input_target_nodes_inputmeta_map,
    std::unordered_map</*child node*/ GradNodeBase*,
                       /*father nodes*/ std::unordered_set<GradNodeBase*>>*
        depending_nodes,
    std::unordered_set<GradNodeBase*>* potential_stop_nodes,
    std::unordered_set<GradNodeBase*>* potential_startup_nodes) {
  if (input_target_nodes_inputmeta_map->empty()) return;

  VLOG(6) << "Runing In GetGraphInfoBetweenTargets";

  // Calculate in_degree for each node
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

    // Check node is target_nodes or not, if node is not target_node,
    // all the next_node will be marked in potential_stop_nodes
    bool is_potential_stop_nodes =
        input_target_nodes_inputmeta_map->count(node);

    // Find and append next nodes
    const std::vector<std::vector<Edge>>& edges = node->GetEdges();
    for (const auto& edge_list : edges) {
      for (const Edge& edge : edge_list) {
        GradNodeBase* next_node = edge.GetMutableGradNode().get();

        // Next node could be nullptr if it is leaf tensor with no
        // AccumulationNode attached
        // Or it could also originated from dispensable inputs
        if (!next_node) continue;

        // if node not in input_target_nodes,
        // all the next_nodes of current node will be inserted to
        // potential_stop_node
        if (is_potential_stop_nodes) {
          potential_stop_nodes->emplace(next_node);
        }

        // Update in_degree
        if (!node_in_degree_map.count(next_node))
          node_in_degree_map[next_node] = 0;
        node_in_degree_map[next_node]++;

        // Record depending relationship
        (*depending_nodes)[next_node].emplace(node);
        queue.push(next_node);
      }
    }
  }
  // Update Graph Info, remove some stop_node in potential_stop_nodes
  UpdateGraphInfo(input_target_nodes_inputmeta_map, depending_nodes,
                  potential_stop_nodes, potential_startup_nodes);
}

void GetTargetNodesInfo(const std::vector<paddle::experimental::Tensor>& inputs,
                        std::unordered_map<GradNodeBase*, AutogradMeta*>*
                            target_nodes_inputmeta_map) {
  VLOG(6) << "Running in GetTargetNodesInfo";
  if (!inputs.empty()) {
    VLOG(6) << "Inputs are not empty";
    size_t num_inputs = inputs.size();
    for (size_t i = 0; i < num_inputs; i++) {
      AutogradMeta* auto_grad_meta =
          EagerUtils::unsafe_autograd_meta(inputs[i]);
      auto target_node = auto_grad_meta->GetMutableGradNode().get();

      PADDLE_ENFORCE_NOT_NULL(target_node,
                              paddle::platform::errors::Fatal(
                                  "There is no grad op for input:%d or it's"
                                  "stop_gradient=True",
                                  i));
      (*target_nodes_inputmeta_map)[target_node] = auto_grad_meta;
    }
  }
}

std::vector<paddle::experimental::Tensor> GetResults(
    const std::vector<paddle::experimental::Tensor>& inputs,
    std::unordered_map<GradNodeBase*, paddle::experimental::Tensor>*
        results_map,
    bool allow_unused, bool create_graph) {
  VLOG(6) << "Running in GetResults";
  if (inputs.empty()) return {};

  std::vector<paddle::experimental::Tensor> results;
  results.reserve(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& input = inputs[i];
    AutogradMeta* auto_grad_meta = EagerUtils::unsafe_autograd_meta(input);
    auto target_node = auto_grad_meta->GetMutableGradNode().get();

    auto iter = results_map->find(target_node);
    if (iter != results_map->end()) {
      // set StopGradient = !create_graph
      AutogradMeta* tensor_auto_grad_meta =
          EagerUtils::autograd_meta(&(iter->second));
      tensor_auto_grad_meta->SetStopGradient(!create_graph);
      results.emplace_back(iter->second);
    } else {
      PADDLE_ENFORCE_EQ(allow_unused, true,
                        paddle::platform::errors::InvalidArgument(
                            "The %d-th input does not appear in the backward "
                            "graph. Please check the input variable or set "
                            "allow_unused=True to get None result.",
                            i));
      results.emplace_back();
    }
  }
  return results;
}

// Enforce GradNode has TensorWrappers as Input
void EnforceGradNodeHasInput(GradNodeBase* node) {
  VLOG(6) << "Running in EnforceGradNodeHasInput";
  PADDLE_ENFORCE_NE(
      node->IsTensorWrappersCleared(), true,
      paddle::platform::errors::Fatal(
          "The TensorWrappers of %s do not exist. This may be because:\n"
          "You calculate backward twice for the same subgraph without "
          "setting retain_graph=True. Please set retain_graph=True in the "
          "first backward/grad call.\n",
          node->name()));
}

// Purify potential_startup_nodes, remove nodes those are the same as
// input_target_nodes
void PurifyPotentialStartUpNodes(
    std::unordered_set<GradNodeBase*>* potential_startup_nodes,
    std::unordered_map<GradNodeBase*, AutogradMeta* /* InputMeta */>*
        input_target_nodes_inputmeta_map) {
  VLOG(6) << "Running in PurifyPotentialStartUpNodes";
  if (input_target_nodes_inputmeta_map->empty()) return;
  std::unordered_set<GradNodeBase*> potential_startup_nodes_to_be_erased;
  for (auto startup_op : *potential_startup_nodes) {
    auto iter = input_target_nodes_inputmeta_map->find(startup_op);
    if (iter != input_target_nodes_inputmeta_map->end()) {
      potential_startup_nodes_to_be_erased.emplace(iter->first);
    }
  }
  if (!potential_startup_nodes_to_be_erased.empty()) {
    for (auto nodes : potential_startup_nodes_to_be_erased) {
      potential_startup_nodes->erase(nodes);
    }
  }
}

std::vector<paddle::experimental::Tensor> RunBackward(
    const std::vector<paddle::experimental::Tensor>& tensors,  // output
    const std::vector<paddle::experimental::Tensor>& grad_tensors,
    bool retain_graph, bool create_graph = false,
    const std::vector<paddle::experimental::Tensor>& inputs = {},
    bool allow_unused = false,
    const std::vector<paddle::experimental::Tensor>& no_grad_vars = {}) {
  VLOG(6) << "Start Backward";
  // *Gradient Hook should happen at node-level
  // *Inplace version check should perform at node-level
  // *Cross-batch accumulation happens at forward pass

  std::unordered_map<GradNodeBase*, AutogradMeta*>
      no_grad_var_nodes_inputmeta_map;
  // Get no_grad_vars's GradNodes and InputMeta Info
  GetTargetNodesInfo(no_grad_vars, &no_grad_var_nodes_inputmeta_map);

  /* --- Initialization --- */
  // 1. Init queue with starting nodes
  // 2. Prepare initial input buffers
  std::queue<GradNodeBase*> queue;
  std::unordered_map<GradNodeBase*, std::unique_ptr<GradTensorHolder>>
      node_input_buffers_dict;
  std::unordered_set<GradNodeBase*> potential_startup_nodes;
  for (size_t i = 0; i < tensors.size(); i++) {
    const paddle::experimental::Tensor& tensor = tensors[i];

    AutogradMeta* auto_grad_meta = EagerUtils::unsafe_autograd_meta(tensor);
    // Get grad input info from target tensors
    auto input_info = auto_grad_meta->OutRankInfo();

    VLOG(2) << "Out Rank of Tensor is slot: " << input_info.first
            << ", rank: " << input_info.second;
    // Get target GradNodeBase from target tensors
    auto shared_grad_node = auto_grad_meta->GetMutableGradNode();

    if (shared_grad_node == nullptr || shared_grad_node.get() == nullptr ||
        auto_grad_meta->StopGradient()) {
      VLOG(3) << "Skip auto grad since there is no grad op for var or loss is "
                 "stop_gradient=True: "
              << tensor.name();
      continue;
    }

    GradNodeBase* grad_node = shared_grad_node.get();

    // Prepare GradTensorHolder
    if (!node_input_buffers_dict.count(grad_node)) {
      VLOG(6) << "Create Value for grad input tensor " << i
              << " of grad node: " << grad_node->name();
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

      if (grad_tensors[i].is_initialized()) {
        // Deep copy
        paddle::experimental::Tensor tmp_tensor;
        tmp_tensor.copy_(grad_tensors[i], true);
        node_input_buffers_dict[grad_node]->add(input_info.first,
                                                input_info.second, tmp_tensor);
      } else {
        node_input_buffers_dict[grad_node]->add(
            input_info.first, input_info.second, grad_tensors[i]);
      }

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

    // Prepare queue, potential startup_nodes
    queue.push(grad_node);
    potential_startup_nodes.emplace(grad_node);
  }

  VLOG(6) << "Update In degree Map for backward";
  // 3. Compute in_degree for each node
  std::unordered_map<GradNodeBase*, int> node_in_degree_map =
      getInDegreeMap(queue);

  // Get input's GradNodes and InputMeta Info
  std::unordered_map<GradNodeBase*, AutogradMeta* /* InputMeta */>
      input_target_nodes_inputmeta_map;
  GetTargetNodesInfo(inputs, &input_target_nodes_inputmeta_map);

  // Purify potential_startup_ops, remove those nodes that are the same as
  // input_target_nodes
  PurifyPotentialStartUpNodes(&potential_startup_nodes,
                              &input_target_nodes_inputmeta_map);

  // Get Graph Info Betweent input target gradnode and outputs
  // Record the depending_nodes and potential_stop_nodes
  std::unordered_map<GradNodeBase* /* child node */,
                     std::unordered_set<GradNodeBase*> /* father node */>
      depending_nodes;
  std::unordered_set<GradNodeBase*> potential_stop_nodes;
  // std::unordered_set<GradNodeBase*> startup_ops;

  GetGraphInfoBetweenTargets(queue, &input_target_nodes_inputmeta_map,
                             &depending_nodes, &potential_stop_nodes,
                             &potential_startup_nodes);

  // ready_queue store all startup nodes
  std::queue<GradNodeBase*> ready_queue;
  // startup op's indegree should be 0
  for (auto node : potential_startup_nodes) {
    if (node_in_degree_map[node] == 0) {
      ready_queue.emplace(node);
    }
  }

  VLOG(1) << " startup_ops' size is :" << ready_queue.size();

  std::unordered_map<GradNodeBase*, paddle::experimental::Tensor> results_map;

  // read_queue is empty only when 1.input equals to output. 2.input can not
  // reach to output.
  if (ready_queue.size() == 0) {
    for (auto input_target_node : input_target_nodes_inputmeta_map) {
      // out rank_info of forward op
      auto rank_info = input_target_node.second->OutRankInfo();
      if (node_input_buffers_dict[input_target_node.first]) {
        auto& target_result =
            node_input_buffers_dict[input_target_node.first]
                ->Buffers()[rank_info.first][rank_info.second];
        // save the target result
        results_map[input_target_node.first] = target_result;
      }
    }
  }

  /* --- Topological Visit --- */
  // 1. Pop queue
  // 2. Run node
  //    |- Check and capture target result
  //    |- node(grads)
  //    |- Prepare for next node
  // 3. Update queue
  VLOG(6) << "Run Backward";
  while (!ready_queue.empty()) {
    GradNodeBase* node = ready_queue.front();
    VLOG(6) << "Running GradNode:" << node->name();
    ready_queue.pop();

    // Run node: This is where Hook happens
    PADDLE_ENFORCE(
        node_input_buffers_dict.count(node),
        paddle::platform::errors::Fatal(
            "Unable to find next node in the GradTensorHolder \n"
            "Trying to run Node without configuring its GradTensorHolder"));

    std::unique_ptr<GradTensorHolder> node_input_buffer =
        std::move(node_input_buffers_dict[node]);

    // get target grad_var from node_input_buffer by inputmeta
    if (input_target_nodes_inputmeta_map.find(node) !=
        input_target_nodes_inputmeta_map.end()) {
      VLOG(6) << "Get target result by by inputmeta";
      // out rank_info of forward op
      auto rank_info = input_target_nodes_inputmeta_map[node]->OutRankInfo();
      // rank_info is a pair, first means slot_id, second means rank.
      auto& target_result =
          node_input_buffer->Buffers()[rank_info.first][rank_info.second];
      // save the target result
      results_map[node] = target_result;
    }

    // no_grad_vars
    if (no_grad_var_nodes_inputmeta_map.find(node) !=
        no_grad_var_nodes_inputmeta_map.end()) {
      VLOG(6) << "Change the input buffer[slot][rank] by Zeros";
      auto rank_info = no_grad_var_nodes_inputmeta_map[node]->OutRankInfo();
      node_input_buffer->SetBufferSlotRankZeros(rank_info.first,
                                                rank_info.second);
    }

    VLOG(6) << "Running GradNode:" << node->name();

    // check input
    EnforceGradNodeHasInput(node);

    VLOG(6) << "Run Backward Kernel with GradTensorHolder";
    // Run Pre Backward Node and get outputs
    std::vector<std::vector<paddle::experimental::Tensor>> grad_output_tensors =
        (*node)(node_input_buffer->Buffers(), create_graph);

    // retain_grad or not
    if (!retain_graph) {
      VLOG(6)
          << "retain_graph is false, need to clear the TensorWrapper of nodes.";
      node->ClearTensorWrappers();
    }

    // TODO(jiabin): Should we erase it or find a more efficient way.
    node_input_buffers_dict.erase(node);

    // Prepare GradTensorHolder for next node
    const std::vector<std::vector<Edge>>& edges = node->GetEdges();

    PADDLE_ENFORCE(edges.size() == grad_output_tensors.size() || edges.empty(),
                   paddle::platform::errors::Fatal(
                       "Number of edges should be either empty ( for leaf node "
                       ") or the same as number of output grad tensors, but we "
                       "got edges size is: %d, grad_output size is: %d",
                       edges.size(), grad_output_tensors.size()));

    for (size_t i = 0; i < edges.size(); i++) {
      for (size_t j = 0; j < edges[i].size(); j++) {
        const Edge& edge = edges[i][j];
        auto edge_rank = edge.GetEdgeRankInfo();
        // Since we make edge has as same rank as bwd outputs, we indexing them
        // with
        // the same rank(i, j)
        auto next_node_shared = edge.GetMutableGradNode();

        // Next node could be nullptr if it is leaf tensor with no
        // AccumulationNode attached
        // Or it could also originated from dispensable inputs
        if (!next_node_shared || !next_node_shared.get() ||
            grad_output_tensors[i].empty()) {
          continue;
        }
        PADDLE_ENFORCE_LT(
            j, grad_output_tensors[i].size(),
            paddle::platform::errors::Fatal(
                "Rank of grad_output_tensors should be less than "
                "grad_output_tensors[i].size(), which is: %d. This error may "
                "indicate autoprune or autograd api error. ",
                grad_output_tensors.size()));
        paddle::experimental::Tensor& grad_output_tensor =
            grad_output_tensors[i][j];

        if ((!grad_output_tensor.defined() ||
             !grad_output_tensor.initialized())) {
          VLOG(6) << "We get grad_output_tensor with slot: " << i
                  << ", rank: " << j << " as uninitialized or undefined tensor";
        }
        VLOG(6) << "Get Edge and grad_output_tensor with slot: " << i
                << ", rank: " << j
                << " 's name is: " << grad_output_tensor.name();

        auto* next_node = next_node_shared.get();
        if (!node_input_buffers_dict.count(next_node)) {
          const auto& input_meta = next_node->InputMeta();
          auto grad_tensor_holder =
              std::make_unique<GradTensorHolder>(input_meta);
          VLOG(6) << "Construct GradTensorHolder for grad node: "
                  << next_node->name();
          node_input_buffers_dict[next_node] = std::move(grad_tensor_holder);
        }
        VLOG(6) << "Sum grad inputs for edge slot: " << edge_rank.first
                << ", rank: " << edge_rank.second;
        node_input_buffers_dict[next_node]->add(
            edge_rank.first, edge_rank.second, grad_output_tensor);

        // Update queue
        node_in_degree_map[next_node]--;

        PADDLE_ENFORCE(
            node_in_degree_map[next_node] >= 0,
            paddle::platform::errors::Fatal(
                "Detected in-degree value smaller than zero. For Node: %s"
                "Node's in-degree cannot be negative",
                next_node->name()));

        bool is_potential_stop_node = potential_stop_nodes.count(next_node);

        if (node_in_degree_map[next_node] == 0 && !is_potential_stop_node) {
          ready_queue.emplace(std::move(next_node));
        }
      }
    }
  }

  return GetResults(inputs, &results_map, allow_unused, create_graph);
}

void Backward(
    const std::vector<paddle::experimental::Tensor>& tensors,  // output
    const std::vector<paddle::experimental::Tensor>& grad_tensors,
    bool retain_graph) {
  VLOG(6) << "Run in Backward";
  RunBackward(tensors, grad_tensors, retain_graph);
}

std::vector<paddle::experimental::Tensor> Grad(
    const std::vector<paddle::experimental::Tensor>& tensors,  // output
    const std::vector<paddle::experimental::Tensor>& inputs,
    const std::vector<paddle::experimental::Tensor>& grad_tensors,
    bool retain_graph, bool create_graph, bool only_inputs, bool allow_unused,
    const std::vector<paddle::experimental::Tensor>& no_grad_vars) {
  VLOG(6) << "Run in Grad";
  return RunBackward(tensors, grad_tensors, retain_graph, create_graph, inputs,
                     allow_unused, no_grad_vars);
}
}  // namespace egr
