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
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/utils.h"

namespace egr {
class TensorWrapper {
 public:
  TensorWrapper() = default;
  explicit TensorWrapper(const paddle::experimental::Tensor& tensor,
                         bool full_reserved = false) {
    /**
     * Normally, we should fully reserved all non-output or non-leaf fwd tensor
     * here. And for fwd output tensor, we should not reserve its autogradmeta,
     * to avoid recursive depends on GradNodeBase
     * **/
    full_reserved_ = full_reserved;
    if (full_reserved_) {
      VLOG(6) << "Fully reserved tensor: " << tensor.name();
      intermidiate_tensor_ = tensor;
      return;
    }

    // shallow copy tensor_impl here
    intermidiate_tensor_.set_impl(tensor.impl());
    intermidiate_tensor_.set_name(tensor.name() + "@Saved");
    PADDLE_ENFORCE_NOT_NULL(
        EagerUtils::unsafe_autograd_meta(tensor),
        paddle::platform::errors::Fatal(
            "Full reserved Tensor should not have null autograd meta, since "
            "tensor_wrapper is used to build backward info. There is no way "
            "for us to build it with null autograd_meta."));
    // copy output_rank
    out_rank_info_ = EagerUtils::OutRankInfo(tensor);
  }

  paddle::experimental::Tensor recover(
      const std::shared_ptr<GradNodeBase>& grad_node) {
    VLOG(6) << "Recover tensor: " << intermidiate_tensor_.name()
            << " for wrapper";
    if (!intermidiate_tensor_.defined()) {
      VLOG(6) << "Return NULL tensor Here. ";
      return paddle::experimental::Tensor();
    }

    // if it's full_reserved just return the full copy of tensor
    if (full_reserved_) {
      return intermidiate_tensor_;
    } else {
      std::shared_ptr<GradNodeBase> new_grad_node = grad_node;
      auto p_ab_autograd_meta =
          std::make_shared<AutogradMeta>(Edge(new_grad_node, out_rank_info_));
      intermidiate_tensor_.set_autograd_meta(
          std::static_pointer_cast<paddle::experimental::AbstractAutogradMeta>(
              p_ab_autograd_meta));
      return intermidiate_tensor_;
    }
  }

 private:
  bool full_reserved_ = false;
  std::pair<size_t, size_t> out_rank_info_;
  paddle::experimental::Tensor intermidiate_tensor_;
};
}  // namespace egr
