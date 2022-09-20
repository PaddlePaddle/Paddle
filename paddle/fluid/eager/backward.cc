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

#include "paddle/fluid/eager/general_grad.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"

namespace egr {

std::unordered_map<GradNodeBase*, int> getInDegreeMap(
    const std::deque<GradNodeBase*>& init_queue) {
  // Calculate in_degree for each node
  // We can completely remove this pass, if in_degree were set during forward
  // pass
  std::unordered_map<GradNodeBase*, int> node_in_degree_map;

  // Copy nodes
  std::deque<GradNodeBase*> queue = init_queue;
  std::unordered_set<GradNodeBase*> visited;

  // Visit each node exactly once in any order
  while (!queue.empty()) {
    GradNodeBase* node = queue.front();
    queue.pop_front();

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
    const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
        metas = node->OutputMeta();
    for (const auto& meta_list : metas) {
      for (const GradSlotMeta& meta : meta_list) {
        const auto& edge = meta.GetEdge();
        GradNodeBase* next_node = edge.GetMutableGradNode().get();
        // Next node could be nullptr if it is leaf tensor with no
        // AccumulationNode attached
        // Or it could also originated from dispensable inputs
        if (!next_node) continue;

        // Update in_degree
        if (!node_in_degree_map.count(next_node))
          node_in_degree_map[next_node] = 0;
        node_in_degree_map[next_node]++;
        queue.push_back(next_node);
      }
    }
  }

  return node_in_degree_map;
}

// Enforce GradNode has TensorWrappers as Input
void EnforceGradNodeHasInput(GradNodeBase* node) {
  PADDLE_ENFORCE_NE(
      node->IsTensorWrappersCleared(),
      true,
      paddle::platform::errors::Fatal(
          "The TensorWrappers of %s do not exist. This may be because:\n"
          "You calculate backward twice for the same subgraph without "
          "setting retain_graph=True. Please set retain_graph=True in the "
          "first backward/grad call.\n",
          node->name()));
}

void DuplicateCheck(const std::vector<paddle::experimental::Tensor>& inputs,
                    bool is_input) {
  std::unordered_set<AutogradMeta*> visisted_ins;
  std::string msg = is_input ? "inputs" : "outputs";
  for (auto in : inputs) {
    AutogradMeta* auto_grad_meta = EagerUtils::unsafe_autograd_meta(in);
    PADDLE_ENFORCE_EQ(
        visisted_ins.count(auto_grad_meta),
        0,
        paddle::platform::errors::AlreadyExists(
            "%s contain duplicate tensor %s, please check %s carefully.",
            msg,
            in.name(),
            msg));
    visisted_ins.insert(auto_grad_meta);
  }
}

GeneralGrad* GeneralGrad::general_grad_ = new GeneralGrad();

std::vector<paddle::experimental::Tensor> RunBackward(
    const std::vector<paddle::experimental::Tensor>& tensors,  // output
    const std::vector<paddle::experimental::Tensor>& grad_tensors,
    bool retain_graph,
    bool create_graph = false,
    const std::vector<paddle::experimental::Tensor>& inputs = {},
    bool allow_unused = false,
    const std::vector<paddle::experimental::Tensor>& no_grad_vars = {}) {
  VLOG(3) << "Start Backward";

  // *Gradient Hook should happen at node-level
  // *Inplace version check should perform at node-level
  // *Cross-batch accumulation happens at forward pass

  // GeneralGrad
  bool is_general_grad = !inputs.empty();
  if (is_general_grad) GeneralGrad::Instance().Clear();

  /* --- Initialization --- */
  // 1. Init queue with starting nodes
  // 2. Prepare initial input buffers
  std::deque<GradNodeBase*> queue;
  std::deque<GradNodeBase*> orig_queue;
  std::unordered_map<GradNodeBase*, std::unique_ptr<GradTensorHolder>>
      node_input_buffers_dict;
  for (size_t i = 0; i < tensors.size(); i++) {
    const paddle::experimental::Tensor& tensor = tensors[i];

    AutogradMeta* auto_grad_meta = EagerUtils::nullable_autograd_meta(tensor);
    if (auto_grad_meta == nullptr) {
      VLOG(5) << "Skip auto grad since there is no grad op for var or loss is "
                 "stop_gradient=True: "
              << tensor.name();
      continue;
    }
    // Get grad input info from target tensors
    auto input_info = auto_grad_meta->OutRankInfo();

    VLOG(5) << "Out Rank of Tensor is slot: " << input_info.first
            << ", rank: " << input_info.second;
    // Get target GradNodeBase from target tensors
    auto shared_grad_node = auto_grad_meta->GetMutableGradNode();

    if (shared_grad_node == nullptr || shared_grad_node.get() == nullptr ||
        auto_grad_meta->StopGradient()) {
      VLOG(5) << "Skip auto grad since there is no grad op for var or loss is "
                 "stop_gradient=True: "
              << tensor.name();
      continue;
    }

    // TODO(zhanlve): Copy and Modify GradNode if is_general_grad
    GradNodeBase* grad_node = shared_grad_node.get();
    if (is_general_grad) {
      // Save orig grad node
      orig_queue.push_back(grad_node);

      // Replace grad_node with copied grad_node
      grad_node = GeneralGrad::Instance().CopyGradNode(shared_grad_node);

      // Record potential startup grad node
      GeneralGrad::Instance().GetPotentialStartupNodes()->insert(grad_node);
    }

    // Prepare GradTensorHolder
    if (!node_input_buffers_dict.count(grad_node)) {
      VLOG(5) << "Create Value for grad input tensor " << i
              << " of grad node: " << grad_node->name();
      node_input_buffers_dict[grad_node] =
          std::make_unique<GradTensorHolder>(grad_node->InputMeta());
    }
    bool copy_from_grad_t =
        grad_tensors.size() > 0 && grad_tensors[i].initialized();
    if (copy_from_grad_t) {
      PADDLE_ENFORCE(
          grad_tensors.size() == tensors.size(),
          paddle::platform::errors::Fatal(
              "Detected size mismatch between tensors and grad_tensors"
              "grad_tensors should either have "
              "size = 0 or same size as tensors."));
      // Feed given tensor if it's provided
      VLOG(3) << "Fill grad input tensor " << i << "with give grad tensor";

      // Deep copy
      node_input_buffers_dict[grad_node]->CopyValueFromTensor(
          input_info.first, input_info.second, grad_tensors[i]);
    } else {
      VLOG(3) << "Fill grad input tensor " << i << " with 1.0";
      // Initialize tensor with 1.0
      // Forward Tensor "tensor" is passed to indicate tensortype, datatype and
      // dims
      // GradTensorHolder will initialize another tensor with same tensortype,
      // datatype and dims but filled with 1.0
      node_input_buffers_dict[grad_node]->CopyValueFromTensor(
          input_info.first, input_info.second, tensor, /*fill_one=*/true);
    }

    // Prepare queue, potential startup_nodes
    queue.push_back(grad_node);
  }

  if (is_general_grad) {
    // Prepare several vital preprocess for GeneralGrad
    GeneralGrad::Instance().PreparedForGeneralGrad(
        inputs, no_grad_vars, orig_queue, &queue, node_input_buffers_dict);
  }

  VLOG(5) << "Update In degree Map for backward";
  // 3. Compute in_degree for each node
  std::unordered_map<GradNodeBase*, int> node_in_degree_map =
      getInDegreeMap(queue);

  VLOG(5) << "Startup_ops's size is " << queue.size();

  /* --- Topological Visit --- */
  // 1. Pop queue
  // 2. Run node
  //    |- Check and capture target result
  //    |- node(grads)
  //    |- Prepare for next node
  // 3. Update queue
  while (!queue.empty()) {
    GradNodeBase* node = queue.front();
    VLOG(3) << "Preparing GradNode:" << node->name() << " addr:" << node;
    VLOG(4) << EagerUtils::GradNodeStr(*node);
    paddle::platform::RecordEvent node_record_event(
        std::string((*node).name()),
        paddle::platform::TracerEventType::Operator,
        1);

    if (queue.size() > 1 && node_in_degree_map[node] != 0) {
      queue.pop_front();
      continue;
    }
    queue.pop_front();

    // Run node: This is where Hook happens
    auto node_input_buffer_iter = node_input_buffers_dict.find(node);
    PADDLE_ENFORCE_NE(
        node_input_buffer_iter,
        node_input_buffers_dict.end(),
        paddle::platform::errors::Fatal(
            "Unable to find next node in the GradTensorHolder \n"
            "Trying to run Node without configuring its GradTensorHolder."));

    std::unique_ptr<GradTensorHolder> node_input_buffer =
        std::move(node_input_buffer_iter->second);

    // Check input
    EnforceGradNodeHasInput(node);

    VLOG(7) << "Run Backward Kernel with GradTensorHolder.";
    // Run Pre Backward Node and get outputs
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         kSlotSmallVectorSize>
        grad_output_tensors = (*node)(
            node_input_buffer->Buffers(), create_graph, is_general_grad);

    if (!inputs.empty() && is_general_grad) {
      GeneralGrad::Instance().SetResultForEnddingNodes(grad_output_tensors,
                                                       node);
    }

    // retain_grad or not
    if (!retain_graph) {
      VLOG(3)
          << "retain_graph is false, need to clear the TensorWrapper of nodes.";
      node->ClearTensorWrappers();
    }

    // TODO(jiabin): Should we erase it or find a more efficient way.
    node_input_buffers_dict.erase(node_input_buffer_iter);

    // Prepare GradTensorHolder for next node
    const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
        metas = node->OutputMeta();
    PADDLE_ENFORCE(metas.size() == grad_output_tensors.size() || metas.empty(),
                   paddle::platform::errors::Fatal(
                       "Number of edges should be either empty ( for leaf node "
                       ") or the same as number of output grad tensors, but we "
                       "got edges size is: %d, grad_output size is: %d",
                       metas.size(),
                       grad_output_tensors.size()));

    for (size_t i = 0; i < metas.size(); i++) {
      for (size_t j = 0; j < metas[i].size(); j++) {
        const Edge& edge = metas[i][j].GetEdge();
        if (!edge.IsInitialized()) {
          continue;
        }
        auto edge_rank = edge.GetEdgeRankInfo();
        // Since we make edge has as same rank as bwd outputs, we indexing them
        // with the same rank(i, j)
        auto next_node_shared = edge.GetMutableGradNode();
        VLOG(3) << "Node: " << node->name() << " addr:" << node
                << ", Found pending node: " << next_node_shared->name()
                << " addr: " << next_node_shared.get();
        // Next node could be nullptr if it is leaf tensor with no
        // AccumulationNode attached
        // Or it could also originated from dispensable inputs
        if (!next_node_shared || !next_node_shared.get() ||
            grad_output_tensors[i].empty()) {
          continue;
        }

        PADDLE_ENFORCE_LT(
            j,
            grad_output_tensors[i].size(),
            paddle::platform::errors::Fatal(
                "Rank of grad_output_tensors should be less than "
                "grad_output_tensors[i].size(), which is: %d. This error may "
                "indicate autoprune or autograd api error. ",
                grad_output_tensors.size()));
        paddle::experimental::Tensor& grad_output_tensor =
            grad_output_tensors[i][j];

        if ((!grad_output_tensor.defined() ||
             !grad_output_tensor.initialized())) {
          VLOG(7) << "We get grad_output_tensor with slot: " << i
                  << ", rank: " << j << " as uninitialized or undefined tensor";
        }

        VLOG(7) << "Get Edge and grad_output_tensor with slot: " << i
                << ", rank: " << j
                << " 's name is: " << grad_output_tensor.name();

        auto* next_node = next_node_shared.get();
        if (!node_input_buffers_dict.count(next_node)) {
          const auto& input_meta = next_node->InputMeta();
          auto grad_tensor_holder =
              std::make_unique<GradTensorHolder>(input_meta);
          VLOG(7) << "Construct GradTensorHolder for grad node: "
                  << next_node->name();
          node_input_buffers_dict[next_node] = std::move(grad_tensor_holder);
        }

        VLOG(3) << "Sum grad inputs for edge slot: " << edge_rank.first
                << ", rank: " << edge_rank.second;

        node_input_buffers_dict[next_node]->add(edge_rank.first,
                                                edge_rank.second,
                                                grad_output_tensor,
                                                create_graph);

        // Update queue
        node_in_degree_map[next_node]--;
        VLOG(7) << next_node->name()
                << " ref_cnt is: " << node_in_degree_map[next_node];

        PADDLE_ENFORCE(
            node_in_degree_map[next_node] >= 0,
            paddle::platform::errors::Fatal(
                "Detected in-degree value smaller than zero. For Node: %s"
                "Node's in-degree cannot be negative.",
                next_node->name()));

        if (is_general_grad) {
          if (node_in_degree_map[next_node] == 0 &&
              GeneralGrad::Instance().IsNeededNodes(next_node)) {
            if (dynamic_cast<egr::GradNodeAccumulation*>(next_node)) {
              queue.push_front(std::move(next_node));
            } else {
              queue.push_back(std::move(next_node));
            }
          }
        } else {
          if (node_in_degree_map[next_node] == 0) {
            if (dynamic_cast<egr::GradNodeAccumulation*>(next_node)) {
              queue.push_front(std::move(next_node));
            } else {
              queue.push_back(std::move(next_node));
            }
          }
        }
      }
    }
  }

  VLOG(7) << "Run Backward Final hook size: "
          << egr::Controller::Instance().FinalBackwardHooks().size();
  for (auto& hook : egr::Controller::Instance().FinalBackwardHooks()) {
    (*hook)();
  }
  egr::Controller::Instance().ClearFinalBackwardHooks();
  if (!is_general_grad) return {};
  return GeneralGrad::Instance().GetResults(inputs, allow_unused, create_graph);
  VLOG(3) << "Finish Backward";
}

void Backward(
    const std::vector<paddle::experimental::Tensor>& tensors,  // outputs
    const std::vector<paddle::experimental::Tensor>& grad_tensors,
    bool retain_graph) {
  VLOG(3) << "Run in Backward";
  paddle::platform::RecordEvent backward_record_event(
      "backward", paddle::platform::TracerEventType::UserDefined, 1);
  RunBackward(tensors, grad_tensors, retain_graph);
  phi::autotune::AutoTuneStatus::Instance().Update();
}

std::vector<paddle::experimental::Tensor> Grad(
    const std::vector<paddle::experimental::Tensor>& tensors,  // outputs
    const std::vector<paddle::experimental::Tensor>& inputs,
    const std::vector<paddle::experimental::Tensor>& grad_tensors,
    bool retain_graph,
    bool create_graph,
    bool only_inputs,
    bool allow_unused,
    const std::vector<paddle::experimental::Tensor>& no_grad_vars) {
  VLOG(3) << "Run in Grad";

  DuplicateCheck(inputs, true /* is_input */);
  DuplicateCheck(tensors, false /* is_input */);

  return RunBackward(tensors,
                     grad_tensors,
                     retain_graph,
                     create_graph,
                     inputs,
                     allow_unused,
                     no_grad_vars);
}
}  // namespace egr
