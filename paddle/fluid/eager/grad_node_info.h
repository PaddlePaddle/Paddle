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

#pragma once

#include "paddle/top/api/include/tensor.h"

namespace egr {
/**
 * GradNodeBase is base class of all grad node, which is what should be used by
 * eager execution, we define most of backward autograd members here, and for
 * each Operator, they should hold their onw forward Inputs as TensorWrapper.
 *
 * The GradNodeBase will be held in autograd_meta, and it is also a member of
 * Edge, which indicates the edge of backward graph.
 *
 * TODO:(yangzhanlue) GradNodeBase will also in charge of get the correct input
 * from GradOpDescMaker to GradNodeBase.
 *
 * NOTE:GradNodeBase has a method named run, this method should be overrided by
 * the
 * specific derived class, it will prepare backward inputs and double backward's
 * depends. Then, it will call C++ API of backward kernel functions to finish
 * backward computation.
 *
 * NOTE:GradNodeBase holds its own inputs and Outputs
 *
 * Edge is defined to descripe depend of backward, an Edge is what linked
 * between two
 * node, it should contain a Node and rank of this Node (this is used to
 * indicate which
 * input of grad this edge belong).
 * */
class Edge;
class AutogradMeta;

/**
 * GradSlotMeta is used to Record Forward Tensor info to backward, since paddle
 * has lots of operators
 * whose backward logic is depends on if it has some specific inputs or outputs.
 * So, we need a meta info
 * to record it's needs.
 * **/
class GradSlotMeta {
 public:
  GradSlotMeta() = default;
  void Init(size_t size) {
    size_ = static_cast<int>(size);
    stop_gradient_.resize(size, false);
  }

  bool IsInitialized() const { return size_ != -1; }
  bool IsStopGradient(size_t rank) const { return stop_gradient_[rank]; }
  int Size() const { return size_; }
  void SetStopGradient(size_t rank, bool stop_gradient = true) {
    stop_gradient_.at(rank) = stop_gradient;
  }

 private:
  int size_{-1};
  std::vector<bool> stop_gradient_;
};

class GradNodeBase {
 public:
  GradNodeBase() = default;
  GradNodeBase(size_t bwd_in_slot_num, size_t bwd_out_slot_num);
  // TODO(jiabin): Should we have other constructor here?
  virtual ~GradNodeBase() = default;

  /**
   * operator() designed to contian the real backward execution logic, it should
   * be
   * overrided by derived class defined for each operator. It accepts a vector
   * of
   * Tensor which contains grads input of current operator
   *
   * Note: why we need backward inputs and outputs construct as vector of vector
   * of pt::Tensor?
   * Since all of paddle op composite in form of {"Slot name ", vector<Var>},
   * so, vector of vector
   * is better choice to fit this format.
   * **/
  virtual std::vector<std::vector<pt::Tensor>> operator()(
      const std::vector<std::vector<pt::Tensor>>& grads) = 0;

  /**
   * AddEdges is designed to set input tensors' backward Node as current
   * node's Edges.
   * This method should be call in forward code and for double backward depends
   * computation.
   *
   * This one is called slot by slot
   * **/
  void AddEdges(const std::vector<AutogradMeta*>& metas, size_t slot_id);

  /**
   * GetEdges is designed to get all edges of current node**/
  const std::vector<std::vector<Edge>>& GetEdges() const;

  /**
   * Get Input Meta of current Grad node**/
  const std::vector<GradSlotMeta>& InputMeta() const;
  /**
   * Set bwd ins and outs info with forward vars
   * **/

  void SetMultiGradInMeta(const std::vector<AutogradMeta*>& fwd_out,
                          size_t slot_rank);

  void SetGradInMeta(const AutogradMeta& fwd_out, size_t slot_rank);

  void SetMultiGradOutMeta(const std::vector<AutogradMeta*>& fwd_in,
                           size_t slot_rank);

  void SetGradOutMeta(const AutogradMeta& fwd_in, size_t slot_rank);

  /**
   * Default setters for Grad in/out meta this should be used for same special
   * Node which will not create by user
   * **/
  void SetDefaultGradInOutMeta();
  /**
   * Register GradientHook or ReduceHook
   * **/
  void RegisterGradientHook(
      size_t slot_id, size_t rank,
      const std::function<pt::Tensor(const pt::Tensor&)>& hook);
  void RegisterReduceHook(const std::function<void(void)>& hook);

  /**
   * Apply GradientHook or ReduceHook
   * **/
  inline bool GradientHooksRegistered() { return gradient_hooks_.size() != 0; }
  inline bool ReduceHooksRegistered() { return reduce_hooks_.size() != 0; }

  std::vector<std::vector<pt::Tensor>> ApplyGradientHooks(
      const std::vector<std::vector<pt::Tensor>>& tensors);
  void ApplyReduceHooks();

 private:
  // TODO(jiabin): Use SmallVector instead after merge PR from develop

  // Edges recorded the backward related node info, which indicate all edges
  // linked
  // by this Grad Node.
  // Why we need vector<vector<Edge>>: Edges is as same rank as bwd output.
  std::vector<std::vector<Edge>> adj_edges_;

  // bwd_out_meta_ is used to record Grad output info for backward
  std::vector<GradSlotMeta> bwd_out_meta_;

  // bwd_in_meta_ used to record Grad input info for backward
  std::vector<GradSlotMeta> bwd_in_meta_;
  // Gradient Hooks
  // Customer may register a list of hooks which will be called in order during
  // backward
  // Each entry consists one pair of <out_rank, std::function>
  std::vector<
      std::tuple</* slot id */ size_t, /* rank */ size_t,
                 /* hook */ std::function<pt::Tensor(const pt::Tensor&)>>>
      gradient_hooks_;
  std::vector<std::function<void(void)>> reduce_hooks_;
};

class Edge {
 public:
  // Default constructor for Edges in order to construct it for AutogradMeta
  Edge() : in_slot_id_(0), in_rank_(0), grad_node_(nullptr) {}

  // In real use cases we should create Edge from grad node and input rank which
  // indicate which edge it is.
  // Since we have slot design in operators we will have to locate an edge with
  // slot
  // and rank.
  Edge(const std::shared_ptr<GradNodeBase>& grad_node, size_t in_slot_id,
       size_t in_rank)
      : in_slot_id_(in_slot_id), in_rank_(in_rank), grad_node_(grad_node) {}

  Edge(const std::shared_ptr<GradNodeBase>& grad_node,
       const std::pair</* slot_id */ size_t, /* rank */ size_t>& rank_info)
      : in_slot_id_(rank_info.first),
        in_rank_(rank_info.second),
        grad_node_(grad_node) {}

  GradNodeBase* GetGradNode() const { return grad_node_.get(); }

  std::shared_ptr<GradNodeBase> GetMutableGradNode() const {
    return grad_node_;
  }

  std::pair<size_t, size_t> GetEdgeRankInfo() const {
    return std::make_pair(in_slot_id_, in_rank_);
  }

  void SetEdgeRankInfo(size_t slot_id, size_t in_rank) {
    in_slot_id_ = slot_id;
    in_rank_ = in_rank;
  }

  void SetEdgeRankInfo(
      const std::pair</* slot_id */ size_t, /* rank */ size_t>& edge_rank) {
    in_slot_id_ = edge_rank.first;
    in_rank_ = edge_rank.second;
  }

  // Currently we use grad_node_ to identify if a edge is initialized.
  bool IsInitialized() const { return !grad_node_.get(); }

 private:
  size_t in_slot_id_;
  size_t in_rank_;
  std::shared_ptr<GradNodeBase> grad_node_;
};

/**
 * Input Buffer is designed for backward grad accumulate.
 * Since we will have one output used by multi preceding ops in forward pass,
 * we will meet a problem that we need to accumulate multiple grads into one.
 *
 * InputBuffer should have as same format as forward output **/
class InputBuffer {
 public:
  explicit InputBuffer(const std::vector<GradSlotMeta>& meta) {
    VLOG(7) << "Init InputBuffer with meta size: " << meta.size();
    buffer_.resize(meta.size());
    for (size_t i = 0; i < buffer_.size(); i++) {
      VLOG(7) << "Init InputBuffer with meta rank: " << meta[i].Size();
      buffer_[i].resize(meta[i].Size());
    }
  }

  InputBuffer(const InputBuffer& other) = default;

  explicit InputBuffer(std::vector<std::vector<pt::Tensor>>&& inputs)
      : buffer_(std::move(inputs)) {}

  InputBuffer& operator=(const InputBuffer& other) = default;

  // Create new tensor and copy tensor->impl
  void add(size_t slot_id, size_t rank, const pt::Tensor& t,
           bool fill_one = false);

  const std::vector<pt::Tensor>& operator[](const size_t& pos) {
    return buffer_[pos];
  }

  const std::vector<std::vector<pt::Tensor>>& Buffers() { return buffer_; }

 private:
  std::vector<std::vector<pt::Tensor>> buffer_;
};

}  // namespace egr
