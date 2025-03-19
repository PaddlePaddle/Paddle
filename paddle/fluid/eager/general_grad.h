// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <deque>

#include "glog/logging.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/utils/hook_utils.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/grad_tensor_holder.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/platform/profiler.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

namespace egr {

/*
 * GeneralGrad is Helper class to implement custom grad operation between
 * outputs and inputs.
 *
 * **/
class GeneralGrad {
 public:
  static GeneralGrad& Instance() { return *general_grad_; }

  // Get inputs's / no_grad_vars's GradNodes and InputMeta Info
  void GetTargetNodesInfo(const std::vector<paddle::Tensor>& inputs,
                          bool is_no_grad_vars) {
    std::string msg = is_no_grad_vars ? "no_grad_vars" : "inputs";
    VLOG(6) << "Running in GetTargetNodesInfo.";
    if (!inputs.empty()) {
      VLOG(6) << msg << " are not empty.";
      size_t num_inputs = inputs.size();
      for (size_t i = 0; i < num_inputs; i++) {
        AutogradMeta* auto_grad_meta =
            EagerUtils::unsafe_autograd_meta(inputs[i]);
        PADDLE_ENFORCE_NOT_NULL(
            auto_grad_meta,
            common::errors::Fatal(
                "We got %s:[%d] 's autograd meta is NULL.", msg, i));
        auto* target_node = auto_grad_meta->GetMutableGradNode().get();

        if (orig_to_copied_node_map_.count(target_node)) {
          target_node = orig_to_copied_node_map_[target_node].get();
        } else {
          VLOG(6) << "Unable to find target node in "
                     "orig_to_copied_node_map_, likely indicating an "
                     "unused input";
        }

        PADDLE_ENFORCE_NOT_NULL(
            target_node,
            common::errors::Fatal("There is no grad op for %s:[%d] or it's "
                                  "stop_gradient=True.",
                                  msg,
                                  i));

        if (is_no_grad_vars) {
          (no_grad_var_nodes_inputmeta_map_)[target_node] = auto_grad_meta;
        } else {
          // normal input
          (input_target_nodes_inputmeta_map_)[target_node] = auto_grad_meta;
        }
      }
    }
  }

  // Purify potential_startup_nodes_, remove nodes those are the same as
  // input_target_nodes
  void PurifyPotentialStartUpNodes() {
    VLOG(6) << "Running in PurifyPotentialStartUpNodes";
    if (input_target_nodes_inputmeta_map_.empty()) {
      VLOG(6) << "No input target nodes found, skip.";
      return;
    }
    std::unordered_set<GradNodeBase*> potential_startup_nodes_to_be_erased;
    for (auto startup_node : potential_startup_nodes_) {
      auto iter = input_target_nodes_inputmeta_map_.find(startup_node);
      if (iter != input_target_nodes_inputmeta_map_.end()) {
        potential_startup_nodes_to_be_erased.emplace(iter->first);
      }
    }
    if (!potential_startup_nodes_to_be_erased.empty()) {
      for (auto nodes : potential_startup_nodes_to_be_erased) {
        potential_startup_nodes_.erase(nodes);
      }
    }
  }

  // Update Graph Info and remove some nodes those doesn't need to be
  // stored in potential_startup_nodes_
  void UpdateGraphInfo() {
    std::unordered_set<GradNodeBase*> startup_ops;
    VLOG(6) << "Running in UpdateGraphInfo";
    std::deque<GradNodeBase*> queue;
    for (auto& target_nodes_inputmeta_pair :
         input_target_nodes_inputmeta_map_) {
      queue.push_back(target_nodes_inputmeta_pair.first);
      needed_nodes_.emplace(target_nodes_inputmeta_pair.first);
    }
    std::unordered_set<GradNodeBase*> visited;
    std::unordered_set<GradNodeBase*> input_target_nodes_on_path;
    while (!queue.empty()) {
      auto* target_node = queue.front();
      queue.pop_front();
      if (visited.count(target_node)) {
        continue;
      }
      visited.insert(target_node);
      if (!(depending_nodes_)[target_node].empty()) {
        auto preceding_nodes = (depending_nodes_)[target_node];
        for (auto pre_nodes : preceding_nodes) {
          queue.push_back(pre_nodes);
          needed_nodes_.emplace(pre_nodes);
          if (IsInputTargetNodes(pre_nodes)) {
            input_target_nodes_on_path.emplace(pre_nodes);
          }
        }
      } else {  // startup_ops have no preceding nodes
        VLOG(6) << "Emplace startup_ops";
        startup_ops.emplace(target_node);
        needed_nodes_.emplace(target_node);
      }
    }

    for (auto& target_nodes_inputmeta_pair :
         input_target_nodes_inputmeta_map_) {
      if (!input_target_nodes_on_path.count(
              target_nodes_inputmeta_pair.first)) {
        ending_nodes_.emplace(target_nodes_inputmeta_pair.first);
      }
    }

    // Purify potential_startup_nodes_ again, remove some
    // potential startup nodes that unreach to input target nodes
    if (!startup_ops.empty()) {
      std::unordered_set<GradNodeBase*> potential_startup_nodes_to_be_erased;
      for (auto node : potential_startup_nodes_) {
        if (startup_ops.count(node) == 0) {
          VLOG(6) << "Set up potential_startup_nodes_to_be_erased";
          potential_startup_nodes_to_be_erased.emplace(node);
        }
      }
      if (!potential_startup_nodes_to_be_erased.empty()) {
        for (auto node : potential_startup_nodes_to_be_erased) {
          VLOG(6) << "Erase nodes in potential_startup_nodes_to_be_erased";
          potential_startup_nodes_.erase(node);
        }
      }
    }  // TODO(jiabin): May we need some check here.
  }

  // Get Graph Info Between input target GradNode and outputs,
  // record depending_nodes_
  void GetGraphInfoBetweenTargets(const std::deque<GradNodeBase*>& init_queue) {
    VLOG(6) << "Running In GetGraphInfoBetweenTargets";

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

      // Find and append next nodes
      const paddle::small_vector<std::vector<GradSlotMeta>,
                                 kSlotSmallVectorSize>& metas =
          node->OutputMeta();
      for (const auto& meta_list : metas) {
        for (const GradSlotMeta& meta : meta_list) {
          const auto& edge = meta.GetEdge();
          GradNodeBase* next_node = edge.GetMutableGradNode().get();

          // Next node could be nullptr if it is leaf tensor with no
          // AccumulationNode attached
          // Or it could also originated from dispensable inputs
          if (!next_node) continue;

          // Record depending relationship
          (depending_nodes_)[next_node].emplace(node);
          queue.push_back(next_node);
        }
      }
    }
  }

  void ModifyReadyQueue(std::deque<GradNodeBase*>* queue) {
    std::deque<GradNodeBase*> tmp_queue;
    for (auto nodes : potential_startup_nodes_) {
      tmp_queue.push_back(nodes);
    }
    tmp_queue.swap(*queue);
  }

  // Set result for input target grad_var when potential_startup_nodes_ is empty
  void SetResultForInputTargetVar(
      const std::unordered_map<GradNodeBase*,
                               std::unique_ptr<GradTensorHolder>>&
          node_input_buffers_dict) {
    if (potential_startup_nodes_.size() == 0) {
      for (auto input_target_node : *GetInputTargetNodesInputMetaMap()) {
        // out rank_info of forward op
        auto rank_info = input_target_node.second->OutRankInfo();
        auto iter = node_input_buffers_dict.find(input_target_node.first);
        if (iter != node_input_buffers_dict.end()) {
          auto& target_result =
              (iter->second)->Buffers()[rank_info.first][rank_info.second];
          // save the target result
          results_map_[input_target_node.first] =
              std::make_shared<paddle::Tensor>(target_result);
        }
      }
    }  // TODO(jiabin): Some check here.
  }

  void SetResultForEndingNodes(
      paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
          grad_output,
      GradNodeBase* node) {
    if (IsEndingNodes(node)) {
      VLOG(6) << "Set result for ending_nodes_ with grad_output_tensors";
      results_map_[node] = std::make_shared<paddle::Tensor>(grad_output[0][0]);
    }
  }

  std::shared_ptr<paddle::Tensor> FetchGradForTensor(
      const paddle::Tensor& tensor, egr::GradNodeBase* target_node) {
    std::shared_ptr<paddle::Tensor> tmp{std::make_shared<paddle::Tensor>()};
    VLOG(6)
        << "Running in FetchGradForTensor, prepare FetchGrad Hook for tensor: "
        << tensor.name();
    auto hook = [tmp](const paddle::Tensor& t) {
      auto tmp_grad = tmp.get();
      if (t.defined()) {
        VLOG(6) << "Set impl for FetchGrad Hook for tensor: " << t.name();
        tmp_grad->set_impl(t.impl());
        tmp_grad->set_autograd_meta(t.mutable_autograd_meta());
        return t;
      } else {
        VLOG(6) << "Retain NULL paddle::Tensor in FetchGrad Hook";
        return paddle::Tensor();
      }
    };

    // Append to GradientHooks
    auto rank_info = EagerUtils::unsafe_autograd_meta(tensor)->OutRankInfo();
    target_node->RegisterGradientHook(
        rank_info.first,
        rank_info.second,
        std::make_shared<egr::CppTensorHook>(hook));
    return tmp;
  }

  // Register Hook to fetch input's gradients, when input's grad node is not an
  // ending node in backward graph. If input's grad node is an ending node in
  // backward graph, use grad node's output as inputs' gradients and no need to
  // register Hook. Please note that ending node must be GradNodeAccumulation
  // after ModifyBackwardGraph function.
  void RegisterFetchGradHook(const std::vector<paddle::Tensor>& inputs) {
    VLOG(6) << "Running in RegisterFetchGradHook.";
    if (!inputs.empty()) {
      size_t num_inputs = inputs.size();
      for (size_t i = 0; i < num_inputs; i++) {
        AutogradMeta* auto_grad_meta =
            EagerUtils::unsafe_autograd_meta(inputs[i]);
        auto* target_node = auto_grad_meta->GetMutableGradNode().get();

        if (dynamic_cast<egr::GradNodeAccumulation*>(target_node)) {
          VLOG(6)
              << "No need to call FetchGradForTensor for GradNodeAccumulation";
          continue;
        }

        if (orig_to_copied_node_map_.count(target_node)) {
          target_node = orig_to_copied_node_map_[target_node].get();
          if (copied_node_to_ending_node_map_.count(target_node)) {
            VLOG(6) << "No need to call FetchGradForTensor for ending_nodes";
            continue;
          }
        }

        PADDLE_ENFORCE_NOT_NULL(
            target_node,
            common::errors::Fatal("There is no grad op for inputs:[%d] or it's"
                                  "stop_gradient=True.",
                                  i));

        if (!IsEndingNodes(target_node)) {
          // Fetch grad for tensor in target_node on path.
          auto fetched_grad = FetchGradForTensor(inputs[i], target_node);
          results_map_[target_node] = fetched_grad;
        }
      }
    }
  }

  void SetNodeToAccumulationNode(GradNodeBase* node) {
    if (dynamic_cast<egr::GradNodeAccumulation*>(node)) return;
    if (!(depending_nodes_)[node].empty()) {
      // Find preceding_nodes of current node.
      auto preceding_nodes = (depending_nodes_)[node];
      for (auto pre_nodes : preceding_nodes) {
        paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
            pre_nodes_edges = pre_nodes->MutableOutputMeta();
        for (size_t i = 0; i < pre_nodes_edges.size(); i++) {
          for (size_t j = 0; j < pre_nodes_edges[i].size(); j++) {
            const auto& edge_ = pre_nodes_edges[i][j].GetEdge();
            if (edge_.GetGradNode() == node) {
              Edge& pre_node_edge = pre_nodes_edges[i][j].GetMutableEdge();

              if (copied_node_to_ending_node_map_.count(node)) {
                pre_node_edge.SetGradNode(
                    copied_node_to_ending_node_map_[node]);
              } else {
                auto autograd_meta = egr::AutogradMeta(edge_);
                std::shared_ptr<GradNodeBase> shared_grad_node_accumulation =
                    std::make_shared<egr::GradNodeAccumulation>(&autograd_meta);
                pre_node_edge.SetGradNode(shared_grad_node_accumulation);
                copied_node_to_ending_node_map_[node] =
                    shared_grad_node_accumulation;
              }

              auto* grad_node = pre_node_edge.GetGradNode();
              needed_nodes_.emplace(grad_node);
              ending_nodes_.emplace(grad_node);
              input_target_nodes_inputmeta_map_[grad_node] =
                  input_target_nodes_inputmeta_map_[node];

              VLOG(6)
                  << node->name() << " (addr:" << node
                  << ") has been transformed to GradNodeAccumulation (addr: "
                  << grad_node << ")";

              // Copy Hook func
              if (node->GradientHooksRegistered()) {
                VLOG(6) << "Copy hook func from node: " << node->name()
                        << " (addr: " << node
                        << ") to GradNodeAccumulation (addr: " << grad_node
                        << ")";
                grad_node->SetGradientHookFunctions(
                    node->GetGradientHookFunctions());
              }
            }  // or this node has no need to change
          }
        }
      }
    }
  }

  void ModifyBackwardGraph(std::deque<GradNodeBase*>* queue) {
    std::deque<GradNodeBase*> queue_ = *queue;
    std::unordered_set<GradNodeBase*> visited;

    while (!queue_.empty()) {
      GradNodeBase* node = queue_.front();
      queue_.pop_front();

      if (visited.count(node)) {
        continue;
      }
      visited.insert(node);

      if (IsInputTargetNodes(node) && IsEndingNodes(node)) {
        SetNodeToAccumulationNode(node);
        continue;
      }

      paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
          meta = node->MutableOutputMeta();
      for (size_t i = 0; i < meta.size(); i++) {
        for (size_t j = 0; j < meta[i].size(); j++) {
          Edge& edge = meta[i][j].GetMutableEdge();
          std::shared_ptr<GradNodeBase> next_node = edge.GetMutableGradNode();

          if (!next_node) continue;

          if (no_grad_var_nodes_inputmeta_map_.count(next_node.get()) &&
              (no_grad_var_nodes_inputmeta_map_[next_node.get()]
                   ->OutRankInfo() == edge.GetEdgeRankInfo())) {
            VLOG(3) << "Get no grad edge from grad_node: " << node->name()
                    << " : " << node << " to:" << next_node->name() << ", "
                    << next_node.get() << " with output rank info: "
                    << edge.GetEdgeRankInfo().first << ", "
                    << edge.GetEdgeRankInfo().second;
            // no_grad_var's grad no need to be computed
            meta[i][j].SetStopGradient(true);
            edge.Clear();
            continue;
          }

          if (meta.size() != 1 && IsNeededNodes(node) &&
              !IsNeededNodes(next_node.get()) && !IsEndingNodes(node)) {
            VLOG(3) << "Get stop edge from grad_node: " << node->name() << " : "
                    << node << " to:" << next_node->name() << ", "
                    << next_node.get() << " with output rank info: " << i
                    << ", " << j;
            // No need to compute grad from needed Nodes to no need Nodes
            meta[i][j].SetStopGradient(true);
            edge.Clear();
            continue;
          }

          // Update BFS queue
          queue_.push_back(next_node.get());
        }
      }
    }
  }

  std::vector<paddle::Tensor> GetResults(
      const std::vector<paddle::Tensor>& inputs,
      bool allow_unused,
      bool create_graph) {
    VLOG(6) << "Running in GetResults";
    if (inputs.empty()) return {};

    std::vector<paddle::Tensor> results;
    results.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
      auto& input = inputs[i];
      AutogradMeta* auto_grad_meta = EagerUtils::unsafe_autograd_meta(input);

      auto* target_node = auto_grad_meta->GetMutableGradNode().get();
      if (orig_to_copied_node_map_.count(target_node)) {
        target_node = orig_to_copied_node_map_[target_node].get();
        if (copied_node_to_ending_node_map_.count(target_node)) {
          target_node = copied_node_to_ending_node_map_[target_node].get();
        }
      } else {
        VLOG(6) << "Unable to find target node in "
                   "orig_to_copied_node_map_, likely indicating an unused "
                   "input";
      }
      auto iter = results_map_.find(target_node);
      if (iter != results_map_.end()) {
        // set StopGradient = !create_graph
        AutogradMeta* tensor_auto_grad_meta =
            EagerUtils::autograd_meta(iter->second.get());
        tensor_auto_grad_meta->SetStopGradient(!create_graph);
        results.emplace_back(*(iter->second.get()));
      } else {
        PADDLE_ENFORCE_EQ(allow_unused,
                          true,
                          common::errors::InvalidArgument(
                              "The %d-th input does not appear in the backward "
                              "graph. Please check the input tensor or set "
                              "allow_unused=True to get None result.",
                              i));
        results.emplace_back();
      }
    }
    Clear();
    return results;
  }

  bool IsNeededNodes(GradNodeBase* node) { return needed_nodes_.count(node); }

  bool IsEndingNodes(GradNodeBase* node) { return ending_nodes_.count(node); }

  bool IsInputTargetNodes(GradNodeBase* node) {
    auto iter = input_target_nodes_inputmeta_map_.find(node);
    if (iter != input_target_nodes_inputmeta_map_.end()) {
      return true;
    }
    return false;
  }

  std::unordered_map<GradNodeBase*, AutogradMeta*>*
  GetNoGradVarNodesInputMetaMap() {
    return &no_grad_var_nodes_inputmeta_map_;
  }

  std::unordered_map<GradNodeBase*, AutogradMeta*>*
  GetInputTargetNodesInputMetaMap() {
    return &input_target_nodes_inputmeta_map_;
  }

  std::unordered_set<GradNodeBase*>* GetPotentialStartupNodes() {
    return &potential_startup_nodes_;
  }

  GradNodeBase* CopyGradNode(const std::shared_ptr<GradNodeBase>& orig_node) {
    if (orig_to_copied_node_map_.count(orig_node.get())) {
      return orig_to_copied_node_map_[orig_node.get()].get();
    }
    std::shared_ptr<GradNodeBase> copied_node = orig_node->Copy();

    // Save node and update mapping
    orig_to_copied_node_map_[orig_node.get()] = copied_node;
    copied_grad_nodes_.push_back(copied_node);
    VLOG(3) << "Copied Node: " << orig_node->name() << " ptr: " << orig_node
            << " to ptr: " << copied_node;
    return copied_node.get();
  }

  void CopyBackwardGraph(const std::deque<GradNodeBase*>& orig_init_queue) {
    std::deque<GradNodeBase*> queue = orig_init_queue;
    std::unordered_set<GradNodeBase*> visited;

    // BFS and recursively copy the grad nodes
    while (!queue.empty()) {
      GradNodeBase* orig_node = queue.front();
      queue.pop_front();
      if (visited.count(orig_node)) {
        continue;
      }
      visited.insert(orig_node);

      PADDLE_ENFORCE(
          orig_to_copied_node_map_.count(orig_node),
          common::errors::Fatal(
              "Cannot copy backward graph,"
              "unable to find copied target for certain grad node."));
      GradNodeBase* copied_node = orig_to_copied_node_map_[orig_node].get();

      const paddle::small_vector<std::vector<GradSlotMeta>,
                                 kSlotSmallVectorSize>& orig_meta =
          orig_node->OutputMeta();
      paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
          copied_edges = copied_node->MutableOutputMeta();
      for (size_t i = 0; i < orig_meta.size(); i++) {
        for (size_t j = 0; j < orig_meta[i].size(); j++) {
          const Edge& orig_edge = orig_meta[i][j].GetEdge();
          Edge& copied_edge = copied_edges[i][j].GetMutableEdge();

          std::shared_ptr<GradNodeBase> orig_next_node =
              orig_edge.GetMutableGradNode();
          if (!orig_next_node) continue;

          // Copy Next Node
          std::shared_ptr<GradNodeBase> copied_next_node;
          if (orig_to_copied_node_map_.count(orig_next_node.get())) {
            copied_next_node = orig_to_copied_node_map_[orig_next_node.get()];

          } else {
            copied_next_node = orig_next_node->Copy();
            orig_to_copied_node_map_[orig_next_node.get()] = copied_next_node;
            VLOG(3) << "Copied Node: " << orig_next_node->name()
                    << " ptr: " << orig_next_node
                    << " to ptr: " << copied_next_node;
            copied_grad_nodes_.push_back(copied_next_node);
          }

          // Update Edge's Grad Node
          copied_edge.SetGradNode(copied_next_node);

          // Update BFS queue
          queue.push_back(orig_next_node.get());
        }
      }
    }
  }

  void PreparedForGeneralGrad(
      const std::vector<paddle::Tensor>& inputs,
      const std::vector<paddle::Tensor>& no_grad_vars,
      const std::deque<GradNodeBase*>& orig_queue,
      std::deque<GradNodeBase*>* queue,
      const std::unordered_map<GradNodeBase*,
                               std::unique_ptr<GradTensorHolder>>&
          node_input_buffers_dict) {
    // Copy Backward Graph
    CopyBackwardGraph(orig_queue);
    // Get no_grad_vars's GradNodes and InputMeta Info
    GetTargetNodesInfo(no_grad_vars, true /* is_no_grad_vars */);
    // Get inputs's GradNodes and InputMeta Info
    GetTargetNodesInfo(inputs, false /* is_no_grad_vars */);
    // Purify potentialstartup_ops, remove those nodes that are the same as
    // input_target_nodes
    PurifyPotentialStartUpNodes();
    // Get Graph Info Between input target gradnode and outputs
    // Record the depending_nodes_ and potential_startup_nodes_
    GetGraphInfoBetweenTargets(*queue);
    // Update Graph Info, remove some nodes in
    // potential_startup_nodes_
    UpdateGraphInfo();
    // Reset queue. Queue is empty only when
    // 1.input equals to output. 2.input can not reach to output.
    ModifyReadyQueue(queue);
    // Set result for input target grad_var when queue is empty
    if (queue->empty()) {
      SetResultForInputTargetVar(node_input_buffers_dict);
    } else {
      // TODO(wuweilong): Find a better design here.
      ModifyBackwardGraph(queue);
      // Register Hook to fetch input's gradients
      RegisterFetchGradHook(inputs);
    }
  }

  void Clear() {
    no_grad_var_nodes_inputmeta_map_.clear();
    input_target_nodes_inputmeta_map_.clear();
    potential_startup_nodes_.clear();
    depending_nodes_.clear();
    results_map_.clear();
    copied_grad_nodes_.clear();
    orig_to_copied_node_map_.clear();
    copied_node_to_ending_node_map_.clear();
    needed_nodes_.clear();
    ending_nodes_.clear();
  }

 private:
  GeneralGrad() = default;
  static GeneralGrad* general_grad_;
  // no_grad_vars's GradNode and GradNode's InputMeta.
  std::unordered_map<GradNodeBase*, AutogradMeta* /* InputMeta */>
      no_grad_var_nodes_inputmeta_map_;
  // inputs's GradNode and GradNode's InputMeta.
  std::unordered_map<GradNodeBase*, AutogradMeta* /* InputMeta */>
      input_target_nodes_inputmeta_map_;
  // Record all the potential startup_nodes, will be changed.
  std::unordered_set<GradNodeBase*> potential_startup_nodes_;
  std::unordered_map<GradNodeBase* /* next node */,
                     std::unordered_set<GradNodeBase*> /* pre nodes */>
      depending_nodes_;
  std::unordered_map<GradNodeBase*, std::shared_ptr<paddle::Tensor>>
      results_map_;

  std::vector<std::shared_ptr<GradNodeBase>> copied_grad_nodes_;
  std::unordered_map<GradNodeBase*, std::shared_ptr<GradNodeBase>>
      orig_to_copied_node_map_;
  std::unordered_set<GradNodeBase*> needed_nodes_;
  // Record which grad_node has been transformed to AccumulationNode
  std::unordered_map<GradNodeBase*, std::shared_ptr<GradNodeBase>>
      copied_node_to_ending_node_map_;
  std::unordered_set<GradNodeBase*> ending_nodes_;

  DISABLE_COPY_AND_ASSIGN(GeneralGrad);
};

}  // namespace egr
