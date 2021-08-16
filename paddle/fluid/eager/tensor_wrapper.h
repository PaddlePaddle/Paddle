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

/**
 * We now still need TensorWrapper and it is designed to Copy
 * tensor in autograd mode.
 *
 * Since in autograd usage, we need to pass autograd_meta to
 * backward computation however in tensor interface add to much
 * autograd_related method is not a good choice.
 *
 * In TensorWrapper we will keep autograd info to backward, only
 * for input var, but for output var it will only copy autograd
 * with no grad **/

#pragma once
#include "paddle/fluid/eager/grad_node_info.h"

namespace egr {
class TensorWrapper {
 public:
  explicit TensorWrapper(
      const pt::Tensor& tensor， bool full_reserved = false) {
    /**
     * Normally, we should fully reserved all non-output or non-leaf fwd tensor
     * here. And for fwd output tensor, we should not reserve its autogradmeta,
     * to
     * avoid recursive depends on GradNodeBase
     * **/

    if (full_reserved_) {
      VLOG(0) << "Fully reserved " intermidiate_tensor_ = tensor;
      return;
    }

    // shallow copy tensor_impl here
    intermidiate_tensor_.set_impl(tensor.impl());

    PADDLE_ENFORCE_NOT_NULL(
        tensor.autograd_meta(),
        "Full reserved Tensor should not have null autograd meta");
    // copy output_rank
    output_rank_ = EagerUtils::output_rank(tensor);
  }

  pt::Tensor recover(const std::shared_ptr<GradNodeBase>& grad_node) {
    if (!intermidiate_tensor_.defined()) {
      VLOG(1) << "Return NULL tensor Here. ";
      return Tensor();
    }

    auto grad_node = full_reserved_
                         ? EagerUtils::grad_node(intermidiate_tensor_)
                         : grad_node;

    // if it's full_reserved just return the full copy of tensor
    if (full_reserved_) {
      return intermidiate_tensor_;
    } else {
      auto p_ab_autograd_meta = std::make_shared(Edge(grad_node, output_rank_));
      intermidiate_tensor_.set_autograd_meta(
          std::static_pointer_cast<pt::AbstractAutogradMeta>(
              p_ab_autograd_meta));
      return intermidiate_tensor_;
    }
  }

 private:
  bool full_reserved_;
  int64_t output_rank_;
  pt::Tensor intermidiate_tensor_;
}
}  // namespace egr
