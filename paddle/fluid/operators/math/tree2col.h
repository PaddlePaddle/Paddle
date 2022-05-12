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

#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {
namespace math {
class TreeNode {
 public:
  size_t node;
  explicit TreeNode(size_t node = 0, size_t index = 0, size_t pclen = 0,
                    size_t depth = 0)
      : node(node), index(index), pclen(pclen), depth(depth) {}
  template <typename T>
  T eta_t(T filter_depth) {
    return ((filter_depth - this->depth) / filter_depth);
  }
  template <typename T>
  T eta_l(T filter_depth) {
    T temp;
    if (this->pclen == 1) {
      temp = 0.5;
    } else {
      temp = (this->index - 1.0) / (this->pclen - 1.0);
    }
    return (1.0 - this->eta_t<T>(filter_depth)) * temp;
  }
  template <typename T>
  T eta_r(T filter_depth) {
    return (1.0 - this->eta_t<T>(filter_depth)) *
           (1.0 - this->eta_l<T>(filter_depth));
  }
  TreeNode change_node(size_t v) {
    return TreeNode(v, this->index, this->pclen, this->depth);
  }
  size_t get_node() { return this->node; }
  size_t get_depth() { return this->depth; }

 private:
  size_t index, pclen, depth;
};
class Tree2ColUtil {
 public:
  static std::vector<TreeNode> construct_patch(
      size_t root, int max_depth, const std::vector<std::vector<int>> &tr);

  static void construct_tree(const framework::Tensor &EdgeSet,
                             std::vector<std::vector<int>> *tr,
                             size_t *node_count);
};

template <typename DeviceContext, typename T>
class Tree2ColFunctor {
 public:
  void operator()(const DeviceContext &context,
                  const framework::Tensor &EdgeSet,
                  const framework::Tensor &node_features,
                  framework::Tensor *patch, int max_depth);
};
template <typename DeviceContext, typename T>
class Col2TreeFunctor {
 public:
  void operator()(const DeviceContext &context,
                  const framework::Tensor &EdgeSet,
                  const framework::Tensor &out_grad, framework::Tensor *in_grad,
                  int max_depth);
};
}  // namespace math
}  // namespace operators
}  // namespace paddle
