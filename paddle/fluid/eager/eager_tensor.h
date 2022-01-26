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
// framework deps
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
// pten deps
#include "paddle/pten/api/all.h"
#include "paddle/pten/api/lib/api_declare.h"
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/core/convert_utils.h"
/**
 * This class is used by Eager mode for now. It's painful to do this in Eager
 * Mode, the better
 * choice is to use paddle::experimental::Tensor directly. However, we have a
 * punch of nested kernel code, and
 * they use paddle::framework::Variable in inner logic code. So, we have to
 * provide variable in
 * paddle::framework::ExecutionContext to support it. We should remove this as
 * soon as we finish our latest
 * Pten Lib, and use paddle::experimental::Tensor instead.
 *
 * Note: Keep this class as clean as possible.
 * This class should only support method declared in
 * paddle::experimental::Tensor with access method of
 * paddle::framework::Variable no more members are acceptable.
 * **/

namespace egr {
class EagerTensor final {
 public:
  /* Part 1: Constructors */
  EagerTensor()
      : tensor_(std::make_shared<paddle::experimental::Tensor>()),
        var_(paddle::framework::Variable()) {}
  explicit EagerTensor(const std::string& name)
      : tensor_(std::make_shared<paddle::experimental::Tensor>(name)),
        var_(paddle::framework::Variable()) {}
  /**
   * @description: Use a TensorImpl pointer to construct a Tensor
   * @param {shared_ptr<TensorBase>} tensor_impl
   * @return {Tensor}
   */
  explicit EagerTensor(const std::shared_ptr<pten::TensorBase>& tensor_impl)
      : tensor_(std::make_shared<paddle::experimental::Tensor>(tensor_impl)),
        var_(paddle::framework::Variable()) {}

  EagerTensor(const EagerTensor&) = default;
  EagerTensor(EagerTensor&&) = default;

  /* Part 2: Name access methods */
  /**
   * @description: Return the name of current Tensor.
   * @param None
   * @return {const std::string&}
   */
  const std::string& name() const { return tensor_->name(); }
  /**
   * @description: Set the name of current Tensor.
   * @param {const std::string& name}
   * @return None
   */
  void set_name(const std::string& name) { tensor_->set_name(name); }

  /* Part 3: Dimension, DataType and DataLayout methods */
  /**
   * @description: Return the number of elements of current Tensor.
   * @param None
   * @return {int64_t}
   */
  int64_t numel() const { return tensor_->numel(); }
  /**
   * @description: Return the shape (dimensions) of current Tensor.
   * @param None
   * @return {DDim}
   */
  paddle::framework::DDim shape() const { return tensor_->dims(); }

  /**
   * @description: Return the data type of current Tensor.
   * @param None
   * @return {DataType}
   */
  paddle::experimental::DataType type() const { return tensor_->type(); }

  /**
   * @description: Return the layout of current Tensor.
   * @param None
   * @return {DataLayout}
   */
  paddle::experimental::DataLayout layout() const { return tensor_->layout(); }

  /* Part 3: Device and Backend methods */
  /**
   * @description: Return the place (device) of current Tensor.
   * @param None
   * @return {Place}
   */
  paddle::platform::Place place() const { return tensor_->inner_place(); }

  /**
   * Backend judgment APIs, shield the concept of Backend.
   */
  bool is_cpu() const { return paddle::platform::is_cpu_place(place()); }
  bool is_cuda() const { return paddle::platform::is_gpu_place(place()); }

  /* Part 4: Data Access methods */
  /**
   * @description: Return the implemention of current Tensor.
   * @param None
   * @return {std::shared_ptr<TensorBase>}
   */
  std::shared_ptr<pten::TensorBase> impl() const { return tensor_->impl(); }

  /**
   * @description: Set the implemention of current Tensor.
   * @param {std::shared_ptr<TensorBase>}
   * @return None
   */
  void set_impl(const std::shared_ptr<pten::TensorBase>& impl) {
    tensor_->set_impl(impl);
  }

  // TODO(chenweihang): Whether API Tensor need `data` and `mutable_data`?

  // TODO(chenweihang): slice and split methods use kernels?

  /* Part 5: Status utils methods */
  /**
   * @description: Determine whether it is a meaningful Tensor
   * @param None
   * @return {bool}
   */
  bool defined() const { return tensor_->defined(); }

  /**
   * @description: Determine whether Tensor is initialized
   * @param None
   * @return {bool}
   */
  bool initialized() const { return tensor_->initialized(); }

  bool safe_initialized() const {
    return initialized() || var_.IsInitialized();
  }

  /**
   * @description: Reset the Tensor implementation
   * @param None
   * @return {void}
   */
  void reset() { tensor_->reset(); }

  /**
   * @brief Determine whether tensor is DenseTensor
   *
   * @return true
   * @return false
   */
  bool is_dense_tensor() const { return tensor_->is_dense_tensor(); }

  /**
 * @brief Transfer the current Tensor to the specified device and return.
 *
 * @param place, the target place of which the tensor will copy to.
 * @return Tensor
 */
  // TODO(chenweihang): replace Backend by new Place
  EagerTensor copy_to(pten::Backend backend, bool blocking) const {
    if (Var().IsInitialized()) {
      const_cast<EagerTensor*>(this)->SyncToTensor();
    }
    return EagerTensor(tensor_->copy_to(backend, blocking));
  }

  /**
 * @brief Transfer the source Tensor to current Tensor.
 *
 * @param src, the source Tensor to be copied.
 * @param blocking, Should we copy this in sync way.
 * @return void
 */
  void copy_(const EagerTensor& src, const bool blocking) {
    if (src.Var().IsInitialized()) {
      const_cast<EagerTensor*>(&src)->SyncToTensor();
    }
    if (Var().IsInitialized()) {
      SyncToTensor();
    }
    tensor_->copy_(*(src.tensor_.get()), blocking);
  }
  /* Part 6: Operator overloading */
  EagerTensor& operator=(const EagerTensor& x) & {
    tensor_ = x.tensor_;
    var_ = x.var_;
    return *this;
  }
  EagerTensor& operator=(EagerTensor&& x) & {
    tensor_ = std::move(x.tensor_);
    var_ = std::move(x.var_);
    return *this;
  }

  /* Part 7: Autograd methods */
  paddle::experimental::AbstractAutogradMeta* get_autograd_meta() const {
    return tensor_->get_autograd_meta();
  }
  void set_autograd_meta(
      std::shared_ptr<paddle::experimental::AbstractAutogradMeta>
          autograd_meta) {
    tensor_->set_autograd_meta(autograd_meta);
  }

  /** Part 9: Get framework::Variable from EagerTensor **/
  const paddle::framework::Variable& Var() const { return var_; }

  paddle::framework::Variable* MutableVar() { return &var_; }

  /** Part 10: Sync paddle::framework::Variable with pten::Tensor **/
  void SyncToVar(paddle::framework::proto::VarType_Type type =
                     paddle::framework::proto::VarType::LOD_TENSOR) {
    // Synchronize allocation only once.
    if (!var_.IsInitialized()) {
      // TODO(jiabin): Support selected rows later.
      if (this->initialized()) {
        if (type == paddle::framework::proto::VarType::LOD_TENSOR) {
          auto* framework_tensor =
              var_.GetMutable<paddle::framework::LoDTensor>();
          framework_tensor->Resize(tensor_->dims());
          framework_tensor->set_layout(tensor_->layout());
          // Contruct framework::Tensor from egr::EagerTensor
          auto tensor_dense =
              std::dynamic_pointer_cast<pten::DenseTensor>(tensor_->impl());
          if (tensor_dense && tensor_dense.get()) {
            paddle::experimental::SharesStorage(tensor_dense.get(),
                                                framework_tensor);
          } else {
            PADDLE_THROW(paddle::platform::errors::Fatal(
                "Unrecognized egr::EagerTensor type, only "
                "DenseTensor is supported for now."));
          }
        }
      } else {
        PADDLE_THROW(paddle::platform::errors::Fatal(
            "Can not Sync EagerTensor %s whose "
            "pten::DenseTensor is not initialized!",
            name()));
      }
    }
  }
  /** Part 11: Sync paddle::framework::Variable with pten::Tensor **/
  void SyncToTensor() {
    // Synchronize allocation only once.
    if (var_.IsInitialized()) {
      if (var_.IsType<paddle::framework::LoDTensor>()) {
        SetImplWithLegacyTensor<paddle::framework::LoDTensor,
                                pten::DenseTensor>();
      } else if (var_.IsType<paddle::framework::Tensor>()) {
        SetImplWithLegacyTensor<paddle::framework::Tensor, pten::DenseTensor>();
      } else {
        PADDLE_THROW(
            paddle::platform::errors::Fatal("Unable to fetch underlying tensor "
                                            "from VarBase, only LoDTensor and "
                                            "Tensor are supported for now"));
      }
    } else {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Can not Sync EagerTensor %s whose paddle::framework::Variable is "
          "not initialized!",
          name()));
    }
  }

  void ResetVar(const paddle::framework::Variable& src) { var_ = src; }

  const std::shared_ptr<paddle::experimental::Tensor>& Tensor() const {
    return tensor_;
  }

  void set_tensor(const std::shared_ptr<paddle::experimental::Tensor>& tensor) {
    tensor_ = tensor;
  }

 private:
  template <typename LEGACY_TYPE, typename TYPE>
  void SetImplWithLegacyTensor() {
    const auto& framework_tensor = var_.Get<LEGACY_TYPE>();
    if (defined()) {
      VLOG(8) << "Sync Var to initialized tensor for: " << name();
      paddle::experimental::ReMakePtenDenseTensor(
          framework_tensor, static_cast<pten::DenseTensor*>(impl().get()));
    } else {
      VLOG(8) << "Sync Var to uninitialized tensor for: " << name();
      this->set_impl(std::move(
          paddle::experimental::MakePtenDenseTensor(framework_tensor)));
    }
    var_.Clear();
  }

 private:
  /**
  * @description: Use a pten::Tensor pointer to construct a EagerTensor, never
  * public this!!!!.
  * @param {pten::Tensor} tensor
  * @return {EagerTensor}
  */
  explicit EagerTensor(const paddle::experimental::Tensor& tensor)
      : tensor_(std::make_shared<paddle::experimental::Tensor>(tensor)),
        var_(paddle::framework::Variable()) {}

  std::shared_ptr<paddle::experimental::Tensor> tensor_ = nullptr;
  paddle::framework::Variable var_;
};
}  // namespace egr
