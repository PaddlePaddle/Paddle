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
                         bool full_reserved = false,
                         bool no_need_buffer = false) {
    // set inplace_version_snapshot_ according to tensor's current inplace
    // version.
    if (tensor.impl() && phi::DenseTensor::classof(tensor.impl().get())) {
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(tensor.impl().get());
      auto& inplace_version_counter = dense_tensor->InplaceVersionCounter();
      inplace_version_snapshot_ = inplace_version_counter.CurrentVersion();
    }

    /**
     * Normally, we should fully reserved all non-output or non-leaf fwd tensor
     * here. And for fwd output tensor, we should not reserve its autogradmeta,
     * to avoid recursive depends on GradNodeBase
     * **/
    full_reserved_ = full_reserved;
    no_need_buffer_ = no_need_buffer;
    if (full_reserved_) {
      VLOG(6) << "Fully reserved tensor: " << tensor.name();
      intermidiate_tensor_ = tensor;
      if (no_need_buffer_) {
        if (phi::DenseTensor::classof(tensor.impl().get())) {
          // Only Copy Meta
          phi::DenseTensor* dense_tensor =
              static_cast<phi::DenseTensor*>(tensor.impl().get());
          auto tw_dense_tensor =
              std::make_shared<phi::DenseTensor>(*dense_tensor);
          tw_dense_tensor->clear();
          intermidiate_tensor_.set_impl(tw_dense_tensor);
        } else {
          PADDLE_THROW(paddle::platform::errors::Fatal(
              "Unrecognized tensor type for no_need_buffer feature"));
        }
      }
      return;
    }

    // shallow copy tensor_impl here
    if (no_need_buffer) {
      if (phi::DenseTensor::classof(tensor.impl().get())) {
        // Only Copy Meta
        phi::DenseTensor* dense_tensor =
            static_cast<phi::DenseTensor*>(tensor.impl().get());
        auto tw_dense_tensor = std::make_shared<phi::DenseTensor>();
        tw_dense_tensor->set_meta(dense_tensor->meta());
        intermidiate_tensor_.set_impl(tw_dense_tensor);
      } else {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Unrecognized tensor type for no_need_buffer feature"));
      }
    } else {
      intermidiate_tensor_.set_impl(tensor.impl());
    }

    intermidiate_tensor_.set_name(tensor.name() + "@Saved");

    auto* tensor_autograd_meta = EagerUtils::nullable_autograd_meta(tensor);
    if (tensor_autograd_meta) {
      auto autograd_meta =
          std::make_shared<AutogradMeta>(*tensor_autograd_meta);
      autograd_meta->ResetGradNode();
      intermidiate_tensor_.set_autograd_meta(autograd_meta);
      weak_grad_node_ = tensor_autograd_meta->GetMutableGradNode();
    }
  }

  paddle::experimental::Tensor recover() {
    VLOG(6) << "Recover tensor: " << intermidiate_tensor_.name()
            << " for wrapper";
    if (!intermidiate_tensor_.defined()) {
      VLOG(6) << "Return NULL tensor Here. ";
      return paddle::experimental::Tensor();
    }

    check_inplace_version();

    // if it's full_reserved just return the full copy of tensor
    if (full_reserved_) {
      return intermidiate_tensor_;
    } else {
      paddle::experimental::Tensor recovered_tensor = intermidiate_tensor_;

      std::shared_ptr<GradNodeBase> new_grad_node = weak_grad_node_.lock();
      if (new_grad_node) {
        VLOG(3) << "Recovered TensorWrapper with GradNode "
                << new_grad_node->name() << " addr: " << new_grad_node.get();
      } else {
        VLOG(3) << "Recovered TensorWrapper with Empth GradNode";
      }
      auto* intermediate_autograd_meta =
          EagerUtils::unsafe_autograd_meta(intermidiate_tensor_);
      auto p_ab_autograd_meta =
          std::make_shared<AutogradMeta>(*intermediate_autograd_meta);
      if (new_grad_node) {
        p_ab_autograd_meta->SetGradNode(new_grad_node);
      }
      recovered_tensor.set_autograd_meta(p_ab_autograd_meta);
      return recovered_tensor;
    }
  }

  void check_inplace_version() {
    if (no_need_buffer_) {
      VLOG(6) << "There's no need to check inplace_version because "
                 "no_need_buffer_ is true.";
      return;
    }
    if (intermidiate_tensor_.impl() &&
        phi::DenseTensor::classof(intermidiate_tensor_.impl().get())) {
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(intermidiate_tensor_.impl().get());
      auto& inplace_version_counter = dense_tensor->InplaceVersionCounter();

      uint32_t wrapper_version_snapshot = inplace_version_snapshot_;
      uint32_t tensor_version = inplace_version_counter.CurrentVersion();
      PADDLE_ENFORCE_EQ(
          tensor_version, wrapper_version_snapshot,
          paddle::platform::errors::PermissionDenied(
              "Tensor '%s' used in gradient computation has been "
              "modified by an inplace operation. "
              "Its version is %d but the expected version is %d. "
              "Please fix your code to void calling an inplace operator "
              "after using the Tensor which will used in gradient "
              "computation.",
              intermidiate_tensor_.name(), tensor_version,
              wrapper_version_snapshot));
      VLOG(6) << " The wrapper_version_snapshot of Tensor '"
              << intermidiate_tensor_.name() << "' is [ "
              << wrapper_version_snapshot << " ]";
      VLOG(6) << " The tensor_version of Tensor '"
              << intermidiate_tensor_.name() << "' is [ " << tensor_version
              << " ]";
    }
  }

  void clear() { intermidiate_tensor_.reset(); }

 private:
  bool full_reserved_ = false;
  bool no_need_buffer_ = false;
  paddle::experimental::Tensor intermidiate_tensor_;
  std::weak_ptr<egr::GradNodeBase> weak_grad_node_;
  uint32_t inplace_version_snapshot_ = 0;
};
}  // namespace egr
