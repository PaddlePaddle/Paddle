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
#include "paddle/phi/api/lib/utils/allocator.h"
#ifndef PADDLE_NO_PYTHON
#include "paddle/fluid/eager/hooks.h"
#endif
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"

namespace egr {
class TensorWrapper {
 public:
  TensorWrapper() = default;
  explicit TensorWrapper(const paddle::Tensor& tensor,
                         bool no_need_buffer = false) {
    // set inplace_version_snapshot_ according to tensor's current inplace
    // version.
    if (tensor.has_allocation() && tensor.is_dense_tensor()) {
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(tensor.impl().get());
      auto& inplace_version_counter = dense_tensor->InplaceVersionCounter();
      inplace_version_snapshot_ = inplace_version_counter.CurrentVersion();
    } else if (tensor.has_allocation() && tensor.is_dist_tensor()) {
      phi::DenseTensor* dense_tensor =
          static_cast<phi::distributed::DistTensor*>(tensor.impl().get())
              ->unsafe_mutable_value();
      auto& inplace_version_counter = dense_tensor->InplaceVersionCounter();
      inplace_version_snapshot_ = inplace_version_counter.CurrentVersion();
    }

    /**
     * Normally, we should only save data and part of autograd_meta of fwd
     * tensor, and should not reserve its original grad_node,
     * to avoid recursive and additional depends on GradNodeBase
     * **/
    auto* tensor_autograd_meta = EagerUtils::nullable_autograd_meta(tensor);
    no_need_buffer_ = no_need_buffer;
    // shallow copy tensor_impl here
    if (no_need_buffer) {
      if (phi::DenseTensor::classof(tensor.impl().get())) {
        // Only Copy Meta
        phi::DenseTensor* dense_tensor =
            static_cast<phi::DenseTensor*>(tensor.impl().get());
        // TODO(jiabin): It's not a good idea to set memory size to zero, find
        // another way and change this.
        intermediate_tensor_.set_impl(std::make_shared<phi::DenseTensor>(
            std::make_shared<phi::Allocation>(nullptr, 0, tensor.place()),
            dense_tensor->meta()));
      } else if (phi::distributed::DistTensor::classof(tensor.impl().get())) {
        // Copy Global dims, DistAttr and DenseTensorMeta
        phi::distributed::DistTensor* dist_tensor =
            static_cast<phi::distributed::DistTensor*>(tensor.impl().get());
        auto no_buffer_dist_tensor =
            std::make_shared<phi::distributed::DistTensor>(
                dist_tensor->dims(), dist_tensor->dist_attr());
        *no_buffer_dist_tensor->unsafe_mutable_value() = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, tensor.place()),
            dist_tensor->value().meta());
        intermediate_tensor_.set_impl(no_buffer_dist_tensor);
      } else {
        PADDLE_THROW(common::errors::Fatal(
            "Unrecognized tensor type for no_need_buffer feature"));
      }
    } else {
#ifndef PADDLE_NO_PYTHON
      if (egr::SavedTensorsHooks::GetInstance().IsEnable() &&
          tensor.has_allocation() && tensor.is_dense_tensor()) {
        phi::DenseTensor* dense_tensor =
            static_cast<phi::DenseTensor*>(tensor.impl().get());
        intermediate_tensor_.set_impl(std::make_shared<phi::DenseTensor>(
            std::make_shared<phi::Allocation>(nullptr, 0, tensor.place()),
            dense_tensor->meta()));
        auto pack_hook = egr::SavedTensorsHooks::GetInstance().GetPackHook();
        unpack_hook_ = egr::SavedTensorsHooks::GetInstance().GetUnPackHook();
        packed_value_ = (*pack_hook)(tensor);
      } else if (egr::SavedTensorsHooks::GetInstance().IsEnable() &&
                 tensor.initialized() && tensor.is_dist_tensor()) {
        intermediate_tensor_.set_impl(
            std::make_shared<phi::distributed::DistTensor>(
                tensor.dims(),
                static_cast<phi::distributed::DistTensor*>(tensor.impl().get())
                    ->dist_attr()));
        auto dense_tensor =
            static_cast<phi::distributed::DistTensor*>(tensor.impl().get())
                ->value();
        phi::DenseTensor tmp(
            std::make_shared<phi::Allocation>(nullptr, 0, tensor.place()),
            dense_tensor.meta());
        *(static_cast<phi::distributed::DistTensor*>(
              intermediate_tensor_.impl().get())
              ->unsafe_mutable_value()) = tmp;
        auto pack_hook = egr::SavedTensorsHooks::GetInstance().GetPackHook();
        unpack_hook_ = egr::SavedTensorsHooks::GetInstance().GetUnPackHook();
        packed_value_ = (*pack_hook)(tensor);
      } else {
#endif
        intermediate_tensor_.set_impl(tensor.impl());
#ifndef PADDLE_NO_PYTHON
      }
#endif
    }

    if (VLOG_IS_ON(7)) {
      // TODO(jiabin): This may has server performance issue
      intermediate_tensor_.set_name(tensor.name() + "@Saved");
    }

    if (tensor_autograd_meta) {
      auto autograd_meta =
          std::make_shared<AutogradMeta>(*tensor_autograd_meta);
      autograd_meta->ResetGradNode();
      intermediate_tensor_.set_autograd_meta(autograd_meta);
      weak_grad_node_ = tensor_autograd_meta->GetMutableGradNode();
    }
  }

  paddle::Tensor recover() {
    VLOG(6) << "Recover tensor: " << intermediate_tensor_.name()
            << " for wrapper";
    if (!intermediate_tensor_.defined()) {
      VLOG(6) << "Return NULL tensor Here. ";
      return paddle::Tensor();
    }
#ifndef PADDLE_NO_PYTHON
    if (packed_value_ && unpack_hook_) {
      auto tensor_unpacked = (*unpack_hook_)(packed_value_);
      phi::DenseTensor* src_dense_tensor = nullptr;
      if (tensor_unpacked.is_dense_tensor()) {
        VLOG(6) << "tensor_unpacked is DenseTensor";
        src_dense_tensor =
            static_cast<phi::DenseTensor*>(tensor_unpacked.impl().get());
      } else if (tensor_unpacked.is_dist_tensor()) {
        VLOG(6) << "tensor_unpacked is DistTensor";
        src_dense_tensor = static_cast<phi::distributed::DistTensor*>(
                               tensor_unpacked.impl().get())
                               ->unsafe_mutable_value();
      } else {
        PADDLE_THROW(
            common::errors::Fatal("Unrecognized tensor_unpacked type "
                                  "for egr::TensorWrapper::recover"));
      }

      if (intermediate_tensor_.is_dense_tensor()) {
        VLOG(6) << "intermediate_tensor_ is DenseTensor";
        static_cast<phi::DenseTensor*>(intermediate_tensor_.impl().get())
            ->ResetHolder(src_dense_tensor->Holder());
      } else if (intermediate_tensor_.is_dist_tensor()) {
        VLOG(6) << "intermediate_tensor_ is DistTensor";
        static_cast<phi::distributed::DistTensor*>(
            intermediate_tensor_.impl().get())
            ->unsafe_mutable_value()
            ->ResetHolder(src_dense_tensor->Holder());
      } else {
        PADDLE_THROW(
            common::errors::Fatal("Unrecognized intermediate_tensor_ type for "
                                  "egr::TensorWrapper::recover"));
      }
    } else {
#endif
      check_inplace_version();
#ifndef PADDLE_NO_PYTHON
    }
#endif

    paddle::Tensor recovered_tensor = intermediate_tensor_;

    std::shared_ptr<GradNodeBase> new_grad_node = weak_grad_node_.lock();
    if (new_grad_node) {
      VLOG(7) << "Recovered TensorWrapper with GradNode "
              << new_grad_node->name() << " addr: " << new_grad_node.get();
    } else {
      VLOG(7) << "Recovered TensorWrapper with Empty GradNode";
    }
    auto* intermediate_autograd_meta =
        EagerUtils::nullable_autograd_meta(intermediate_tensor_);

    if (intermediate_autograd_meta) {
      auto p_ab_autograd_meta =
          std::make_shared<AutogradMeta>(*intermediate_autograd_meta);
      if (new_grad_node) {
        p_ab_autograd_meta->SetGradNode(new_grad_node);
      }
      recovered_tensor.set_autograd_meta(p_ab_autograd_meta);
    }

    return recovered_tensor;
  }

  paddle::Tensor get_intermediate_tensor() { return intermediate_tensor_; }

  void clear() { intermediate_tensor_.reset(); }

 private:
  void check_inplace_version() {
    if (no_need_buffer_) {
      VLOG(7) << "There's no need to check inplace_version because "
                 "no_need_buffer_ is true.";
      return;
    }
    if (intermediate_tensor_.impl()) {
      phi::DenseTensor* dense_tensor = nullptr;
      if (phi::DenseTensor::classof(intermediate_tensor_.impl().get())) {
        dense_tensor =
            static_cast<phi::DenseTensor*>(intermediate_tensor_.impl().get());
      } else if (phi::distributed::DistTensor::classof(
                     intermediate_tensor_.impl().get())) {
        dense_tensor = static_cast<phi::distributed::DistTensor*>(
                           intermediate_tensor_.impl().get())
                           ->unsafe_mutable_value();
      } else {
        return;
      }

      auto& inplace_version_counter = dense_tensor->InplaceVersionCounter();

      uint32_t wrapper_version_snapshot = inplace_version_snapshot_;
      uint32_t tensor_version = inplace_version_counter.CurrentVersion();
      PADDLE_ENFORCE_EQ(
          tensor_version,
          wrapper_version_snapshot,
          common::errors::PermissionDenied(
              "Tensor '%s' used in gradient computation has been "
              "modified by an inplace operation. "
              "Its version is %d but the expected version is %d. "
              "Please fix your code to void calling an inplace operator "
              "after using the Tensor which will used in gradient "
              "computation.",
              intermediate_tensor_.name(),
              tensor_version,
              wrapper_version_snapshot));
      VLOG(7) << " The wrapper_version_snapshot of Tensor '"
              << intermediate_tensor_.name() << "' is [ "
              << wrapper_version_snapshot << " ]";
      VLOG(7) << " The tensor_version of Tensor '"
              << intermediate_tensor_.name() << "' is [ " << tensor_version
              << " ]";
    }
  }

 private:
  bool no_need_buffer_ = false;
  paddle::Tensor intermediate_tensor_;
  std::weak_ptr<egr::GradNodeBase> weak_grad_node_;
  uint32_t inplace_version_snapshot_ = 0;
#ifndef PADDLE_NO_PYTHON
  std::shared_ptr<egr::PyObjectHolderBase> packed_value_;
  std::shared_ptr<egr::UnPackHookBase> unpack_hook_;
#else
  std::shared_ptr<void> packed_value_;
  std::shared_ptr<void> unpack_hook_;
#endif
};
}  // namespace egr
