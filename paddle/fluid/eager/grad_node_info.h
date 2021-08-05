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

class GradNodeBase {
 public:
  GradNodeBase() = default;
  virtual ~GradNodeBase() = 0;

  /**
   * operator() designed to contian the real backward execution logic, it should
   * be
   * overrided by derived class defined for each operator. It accepts a vector
   * of
   * Tensor which contains grads input of current operator
   * **/
  virtual std::vector<pt::Tensor> operator()(
      const std::vector<pt::Tensor>& grads) = 0;

  /**
   * AddEdge is designed to set all input tensors' backward Node as current
   * node's Edges.
   * This method should be call in forward code and for double backward depends
   * computation.
   * **/
  void AddEdge(const std::vector<pt::Tensor>& tensors);
  /**
   * GetEdges is designed to get all edges of current node**/
  const std::vector<Edge>& GetEdges() const;

  /**
   * Get Input num of current Grad node**/
  size_t InputNum() const { return input_num_; }

 private:
  // Edges recorded the backward related node info, which indicate all edges
  // linked
  // by this Grad Node.
  std::vector<Edge> adj_edges_;

  // TODO(jiabin): Do we need InputMeta here to indicate input info? Or we just
  // need to know
  // how many inputs do we need to create input buffer
  size_t input_num_;
};

class Edge {
 public:
  // Default constructor for Edges in order to construct it for AutogradMeta
  Edge() : grad_node_(nullptr), input_rank_(-1) {}

  // In real use cases we should create Edge from grad node and input rank which
  // indicate
  // which edge it is.
  Edge(const std::shared_ptr<GradNodeBase>& grad_node, size_t input_rank)
      : grad_node_(grad_node), input_rank_(input_rank) {}

  GradNodeBase* GetGradNode() const { return grad_node_.get(); }

  std::shared_ptr<GradNodeBase> GetMutableGradNode() const {
    return grad_node_;
  }

  size_t GetInputRank() const { return input_rank_; }

  void SetInputRank(size_t input_rank) { input_rank_ = input_rank; }

 private:
  std::shared_ptr<GradNodeBase> grad_node_;
  size_t input_rank_;
};
}  // namespace egr
