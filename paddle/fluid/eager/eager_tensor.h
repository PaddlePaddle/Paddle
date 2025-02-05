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
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
// Phi deps
#include "paddle/common/macros.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/compat/convert_utils.h"

namespace egr {

/**
 * VariableCompatTensor class is used by Eager mode for now. It's painful to
 * do this in Eager Mode, the better choice is to design the special Tensor
 * directly in phi and use it in paddle::Tensor.
 * However, we have some special operators, and they use special input variable
 * type, such as vector<string>, unordered_map<wstring, int>, these type cannot
 * cover by DenseTensor or SparseTensor. So, we have to provide a compatible
 * Tensor type like variable to support these special input type. We should
 * remove this as soon as we finish the ResourceTensor in phi.
 *
 * Note: Keep this class as clean as possible.
 * This class should only support method declared in framework::Variable and
 * necessary overridden methods.
 *
 * Note: This class is only used to support types that cannot be supported by
 * the phi Tensor system temporarily. You CANNOT use this class to handle types
 * such as DenseTensor, SelectedRows, etc.
 **/
class VariableCompatTensor
    : public phi::TensorBase,
      public phi::TypeInfoTraits<phi::TensorBase, VariableCompatTensor> {
 public:
  template <typename T>
  const T& Get() const {
    static_assert(
        paddle::framework::IsRegisteredVarType<T>(),
        "Not registered type. Please register T inside var_type_traits.h");
    PADDLE_ENFORCE_NOT_NULL(
        holder_, common::errors::NotFound("Variable is not initialized."));
    PADDLE_ENFORCE_EQ(
        holder_->Type(),
        paddle::framework::VarTypeTrait<T>::kId,
        common::errors::InvalidArgument(
            "The Variable type must be %s, but the type it holds is %s.",
            paddle::framework::ToTypeName(
                paddle::framework::VarTypeTrait<T>::kId),
            paddle::framework::ToTypeName(holder_->Type())));
    return *static_cast<const T*>(holder_->Ptr());
  }

  bool IsInitialized() const { return holder_ != nullptr; }

  template <typename T>
  T* GetMutable() {
    if (!holder_) {
      holder_.reset(new PlaceholderImpl<T>());
    } else {
      PADDLE_ENFORCE_EQ(
          holder_->Type(),
          paddle::framework::VarTypeTrait<T>::kId,
          common::errors::InvalidArgument(
              "The Variable type must be %s, but the type it holds is %s.",
              paddle::framework::ToTypeName(
                  paddle::framework::VarTypeTrait<T>::kId),
              paddle::framework::ToTypeName(holder_->Type())));
    }
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T>
  bool IsType() const {
    return holder_ &&
           holder_->Type() == paddle::framework::VarTypeTrait<T>::kId;
  }

  void Clear() { holder_.reset(); }

  int Type() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_, common::errors::NotFound("Variable is not initialized."));
    return holder_->Type();
  }

  // necessary overridden methods

  static const char* name() { return "VariableCompatTensor"; }

  ~VariableCompatTensor() override = default;

  int64_t numel() const override {
    PADDLE_THROW(common::errors::Unavailable(
        "VariableCompatTensor does not support `numel` method."));
  }

  const phi::DDim& dims() const override {
    PADDLE_THROW(common::errors::Unavailable(
        "VariableCompatTensor does not support `dims` method."));
  }

  phi::DataType dtype() const override {
    PADDLE_THROW(common::errors::Unavailable(
        "VariableCompatTensor does not support `dtype` method."));
  }

  phi::DataLayout layout() const override {
    PADDLE_THROW(common::errors::Unavailable(
        "VariableCompatTensor does not support `layout` method."));
  }

  const phi::Place& place() const override {
    PADDLE_THROW(common::errors::Unavailable(
        "VariableCompatTensor does not support `place` method."));
  }

  bool valid() const override { return IsInitialized(); }

  bool has_allocation() const override { return IsInitialized(); }

  bool initialized() const override { return IsInitialized(); }

  void* AllocateFrom(phi::Allocator* allocator UNUSED,
                     phi::DataType dtype UNUSED,
                     size_t requested_size UNUSED = 0,
                     bool fake_alloc UNUSED = false) override {
    PADDLE_THROW(common::errors::Unavailable(
        "VariableCompatTensor does not support `AllocateFrom` method."));
  }

 private:
  struct Placeholder {
    virtual ~Placeholder() PADDLE_MAY_THROW {}

    inline int Type() const { return type_; }
    inline const void* Ptr() const { return ptr_; }
    inline void* Ptr() { return ptr_; }

   protected:
    inline void Init(void* p, int type) {
      ptr_ = p;
      type_ = type;
    }

    void* ptr_;
    int type_;
  };

  // Placeholder hides type T, so it doesn't appear as a template
  // parameter of Variable.
  template <typename T>
  struct PlaceholderImpl : public Placeholder {
    static_assert(
        paddle::framework::IsRegisteredVarType<T>(),
        "Not registered type. Please register T inside var_type_traits.h");
    PlaceholderImpl() {
      this->Init(&obj_, paddle::framework::VarTypeTrait<T>::kId);
    }

   private:
    T obj_;
  };

  // pointers to a PlaceholderImpl object indeed.
  std::shared_ptr<Placeholder> holder_;
};

inline bool IsVariableCompatTensor(const paddle::Tensor& tensor) {
  return VariableCompatTensor::classof(tensor.impl().get());
}

/**
 * This class is used by Eager mode for now. It's painful to do this in Eager
 * Mode, the better choice is to use paddle::Tensor directly.
 * However, we have a punch of nested kernel code, and they use
 * paddle::framework::Variable in inner logic code. So, we have to provide
 * variable in paddle::framework::ExecutionContext to support it. We should
 * remove this as soon as we finish our latest Phi Lib, and use
 * paddle::Tensor instead.
 *
 * Note: Keep this class as clean as possible.
 * This class should only support method declared in
 * paddle::Tensor with access method of
 * paddle::framework::Variable no more members are acceptable.
 * **/
class EagerVariable final {
 public:
  /* Default constructor and name constructor should only be used for construct
   * output and in fluid*/
  EagerVariable() = default;

  explicit EagerVariable(const std::string& name) : name_(name) {}

  explicit EagerVariable(const paddle::Tensor& tensor) : name_(tensor.name()) {
    if (tensor.defined()) {
      if (tensor.is_dense_tensor()) {
        ConstructVariableFromTensor<phi::DenseTensor>(tensor);
        src_tensor_ = tensor.impl();
      } else if (tensor.is_selected_rows()) {
        ConstructVariableFromTensor<phi::SelectedRows>(tensor);
      } else if (IsVariableCompatTensor(tensor) &&
                 static_cast<const VariableCompatTensor*>(tensor.impl().get())
                     ->IsType<phi::Vocab>()) {
        ConstructVariableFromCompatTensor<phi::Vocab>(tensor);
      } else if (IsVariableCompatTensor(tensor) &&
                 static_cast<const VariableCompatTensor*>(tensor.impl().get())
                     ->IsType<phi::Strings>()) {
        ConstructVariableFromCompatTensor<phi::Strings>(tensor);
      } else {
        PADDLE_THROW(common::errors::Fatal(
            "Unrecognized egr::EagerVariable type, only "
            "DenseTensor and SelectedRows are supported for now."));
      }
    } else {
      VLOG(6) << "Build Empty EagerVariable with name " << name_;
    }
  }

  ~EagerVariable() {
    if (src_tensor_) {
      auto* framework_tensor = var_.GetMutable<phi::DenseTensor>();
      auto tensor_dense = static_cast<phi::DenseTensor*>(src_tensor_.get());
      if (framework_tensor->memory_size() > 0 &&
          (!phi::is_same_place(framework_tensor->place(),
                               tensor_dense->place()) ||
           framework_tensor->dtype() != tensor_dense->dtype())) {
        tensor_dense->ShareBufferWith(*framework_tensor);
      }
    }
  }

  /** Part 11: Construct paddle::framework::Variable with phi::Tensor **/
  std::shared_ptr<phi::TensorBase> GetTensorBase() {
    // Construct allocation only once.
    if (var_.IsInitialized()) {
      if (var_.IsType<phi::DenseTensor>() || var_.IsType<phi::DenseTensor>()) {
        return SetImplWithLegacyTensor<phi::DenseTensor>();
      } else if (var_.IsType<phi::SelectedRows>()) {
        return SetImplWithLegacyTensor<phi::SelectedRows>();
      } else {
        PADDLE_THROW(
            common::errors::Fatal("Unable to fetch underlying tensor "
                                  "from EagerVariable, only DenseTensor and "
                                  "Tensor are supported for now"));
      }
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "Can not Sync EagerVariable %s whose paddle::framework::Variable is "
          "not initialized!",
          name()));
    }
  }
  const paddle::framework::Variable& Var() const { return var_; }

  paddle::framework::Variable* MutableVar() { return &var_; }

  void ResetVar(const paddle::framework::Variable& src) { var_ = src; }

  const std::string& name() const { return name_; }

  void set_name(const std::string& name) { name_ = name; }

 private:
  template <typename VarType>
  std::shared_ptr<phi::TensorBase> SetImplWithLegacyTensor() {
    const auto& framework_tensor = var_.Get<VarType>();
    VLOG(8) << "Sync Var to tensor for: " << name();
    return std::make_shared<VarType>(framework_tensor);
  }

  template <typename VarType>
  void ConstructVariableFromTensor(const paddle::Tensor& tensor) {
    auto* framework_tensor = var_.GetMutable<VarType>();
    // Construct phi::DenseTensor from egr::EagerVariable
    auto tensor_dense = std::dynamic_pointer_cast<VarType>(tensor.impl());

    PADDLE_ENFORCE_EQ(
        (tensor_dense.get() && tensor_dense),
        true,
        common::errors::Fatal(
            "Tensor %s does not hold phi::SelectedRows or phi::DenseTensor. "
            "Or it holds empty impl, this should not happened since we should "
            "treat all kinds of tensor as what they are.",
            tensor.name()));
    *framework_tensor = *tensor_dense;
    if (tensor.is_dense_tensor()) {
      dynamic_cast<phi::DenseTensor*>(framework_tensor)
          ->set_strides(
              std::dynamic_pointer_cast<phi::DenseTensor>(tensor_dense)
                  ->strides());
    }
  }

  template <typename VarType>
  void ConstructVariableFromCompatTensor(const paddle::Tensor& tensor) {
    auto* framework_holder = var_.GetMutable<VarType>();
    // Construct phi::DenseTensor from egr::EagerVariable
    auto* compat_tensor =
        static_cast<VariableCompatTensor*>(tensor.impl().get());
    PADDLE_ENFORCE_NOT_NULL(
        compat_tensor,
        common::errors::Fatal("Tensor %s holds empty impl, this should not "
                              "happened since we should "
                              "treat all kinds of tensor as what they are.",
                              tensor.name()));
    *framework_holder = compat_tensor->Get<VarType>();
  }

 private:
  std::string name_{""};
  paddle::framework::Variable var_;
  std::shared_ptr<phi::TensorBase> src_tensor_;
};
}  // namespace egr
