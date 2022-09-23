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
#include "paddle/fluid/eager/saved_tensors_hooks.h"
#endif

namespace egr {
class TensorWrapper {
 public:
  TensorWrapper() = default;
  explicit TensorWrapper(const paddle::experimental::Tensor& tensor,
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
        intermidiate_tensor_.set_impl(
            std::move(std::make_shared<phi::DenseTensor>(
                std::make_shared<phi::Allocation>(nullptr, 0, tensor.place()),
                std::move(dense_tensor->meta()))));
      } else {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Unrecognized tensor type for no_need_buffer feature"));
      }
    } else {
#ifndef PADDLE_NO_PYTHON
      if (SavedTensorsHooks::GetInstance().IsEnable() &&
          tensor.is_dense_tensor()) {
        phi::DenseTensor* dense_tensor =
            static_cast<phi::DenseTensor*>(tensor.impl().get());
        intermidiate_tensor_.set_impl(
            std::move(std::make_shared<phi::DenseTensor>(
                std::make_shared<phi::Allocation>(nullptr, 0, tensor.place()),
                dense_tensor->meta())));
        auto pack_hook = SavedTensorsHooks::GetInstance().GetPackHook();
        unpack_hook_ = SavedTensorsHooks::GetInstance().GetUnPackHook();
        packed_value_ = reinterpret_cast<PyObject*>((*pack_hook)(tensor));
      } else {
#endif
        intermidiate_tensor_.set_impl(tensor.impl());
#ifndef PADDLE_NO_PYTHON
      }
#endif
    }

    if (VLOG_IS_ON(7)) {
      // TODO(jiabin): This may has server performance issue
      intermidiate_tensor_.set_name(tensor.name() + "@Saved");
    }

    if (tensor_autograd_meta) {
      auto autograd_meta =
          std::make_shared<AutogradMeta>(*tensor_autograd_meta);
      autograd_meta->ResetGradNode();
      intermidiate_tensor_.set_autograd_meta(autograd_meta);
      weak_grad_node_ = tensor_autograd_meta->GetMutableGradNode();
    }
  }
#ifndef PADDLE_NO_PYTHON
  TensorWrapper(const TensorWrapper& other) {
    no_need_buffer_ = other.no_need_buffer_;
    intermidiate_tensor_ = other.intermidiate_tensor_;
    weak_grad_node_ = other.weak_grad_node_;
    inplace_version_snapshot_ = other.inplace_version_snapshot_;
    packed_value_ = other.packed_value_;
    unpack_hook_ = other.unpack_hook_;
    Py_XINCREF(packed_value_);
  }

  TensorWrapper& operator=(const TensorWrapper& other) {
    no_need_buffer_ = other.no_need_buffer_;
    intermidiate_tensor_ = other.intermidiate_tensor_;
    weak_grad_node_ = other.weak_grad_node_;
    inplace_version_snapshot_ = other.inplace_version_snapshot_;
    packed_value_ = other.packed_value_;
    unpack_hook_ = other.unpack_hook_;
    Py_XINCREF(packed_value_);
    return *this;
  }

  ~TensorWrapper() { Py_XDECREF(packed_value_); }
#endif
  paddle::experimental::Tensor recover() {
    VLOG(6) << "Recover tensor: " << intermidiate_tensor_.name()
            << " for wrapper";
    if (!intermidiate_tensor_.defined()) {
      VLOG(6) << "Return NULL tensor Here. ";
      return paddle::experimental::Tensor();
    }
#ifndef PADDLE_NO_PYTHON
    if (packed_value_ && unpack_hook_) {
      auto tensor_unpacked =
          (*unpack_hook_)(reinterpret_cast<void*>(packed_value_));
      auto src_dense_tensor =
          static_cast<phi::DenseTensor*>(tensor_unpacked.impl().get());
      static_cast<phi::DenseTensor*>(intermidiate_tensor_.impl().get())
          ->ResetHolder(src_dense_tensor->MoveMemoryHolder());
    } else {
#endif
      check_inplace_version();
#ifndef PADDLE_NO_PYTHON
    }
#endif

    paddle::experimental::Tensor recovered_tensor = intermidiate_tensor_;

    std::shared_ptr<GradNodeBase> new_grad_node = weak_grad_node_.lock();
    if (new_grad_node) {
      VLOG(7) << "Recovered TensorWrapper with GradNode "
              << new_grad_node->name() << " addr: " << new_grad_node.get();
    } else {
      VLOG(7) << "Recovered TensorWrapper with Empty GradNode";
    }
    auto* intermediate_autograd_meta =
        EagerUtils::nullable_autograd_meta(intermidiate_tensor_);

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

  paddle::experimental::Tensor get_intermidiate_tensor() {
    return intermidiate_tensor_;
  }

  void clear() { intermidiate_tensor_.reset(); }

 private:
  void check_inplace_version() {
    if (no_need_buffer_) {
      VLOG(7) << "There's no need to check inplace_version because "
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
          tensor_version,
          wrapper_version_snapshot,
          paddle::platform::errors::PermissionDenied(
              "Tensor '%s' used in gradient computation has been "
              "modified by an inplace operation. "
              "Its version is %d but the expected version is %d. "
              "Please fix your code to void calling an inplace operator "
              "after using the Tensor which will used in gradient "
              "computation.",
              intermidiate_tensor_.name(),
              tensor_version,
              wrapper_version_snapshot));
      VLOG(7) << " The wrapper_version_snapshot of Tensor '"
              << intermidiate_tensor_.name() << "' is [ "
              << wrapper_version_snapshot << " ]";
      VLOG(7) << " The tensor_version of Tensor '"
              << intermidiate_tensor_.name() << "' is [ " << tensor_version
              << " ]";
    }
  }

 private:
  bool no_need_buffer_ = false;
  paddle::experimental::Tensor intermidiate_tensor_;
  std::weak_ptr<egr::GradNodeBase> weak_grad_node_;
  uint32_t inplace_version_snapshot_ = 0;
#ifndef PADDLE_NO_PYTHON
  PyObject* packed_value_{nullptr};
  std::shared_ptr<UnPackHookBase> unpack_hook_;
#else
  void* packed_value_{nullptr};
  std::shared_ptr<void> unpack_hook_;
#endif
};
}  // namespace egr
