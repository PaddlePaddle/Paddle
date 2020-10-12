//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

#include "paddle/fluid/framework/var_type_traits.h"

namespace paddle {
namespace framework {

// NOTE(liym27): [ What is VariableInplaceVersion used for? ]
//
// VariableInplaceVersion is a version counter and every Variable has a version
// counter.
// It's used to check whether an inplace operation will result in an incorrect
// gradient calculation.
// Version is incemented when the data of the Variable is modified in place.
class VariableInplaceVersion {
 public:
  explicit VariableInplaceVersion(uint32_t inplace_version = 0)
      : inplace_version_(inplace_version) {}
  bool IsUnique() const { return inplace_version_ == 0; }
  void Bump() { ++inplace_version_; }
  uint32_t CurrentVersion() const { return inplace_version_; }

 private:
  uint32_t inplace_version_;
};

class Variable {
 public:
  template <typename T>
  const T& Get() const {
    static_assert(
        IsRegisteredVarType<T>(),
        "Not registered type. Please register T inside var_type_traits.h");
    PADDLE_ENFORCE_NOT_NULL(
        holder_, platform::errors::NotFound("Variable is not initialized."));
    PADDLE_ENFORCE_EQ(
        holder_->Type(), VarTypeTrait<T>::kId,
        platform::errors::InvalidArgument(
            "The Variable type must be %s, but the type it holds is %s.",
            ToTypeName(VarTypeTrait<T>::kId), ToTypeName(holder_->Type())));
    return *static_cast<const T*>(holder_->Ptr());
  }

  bool IsInitialized() const { return holder_ != nullptr; }

  template <typename T>
  T* GetMutable() {
    if (!holder_) {
      holder_.reset(new PlaceholderImpl<T>());
    } else {
      PADDLE_ENFORCE_EQ(
          holder_->Type(), VarTypeTrait<T>::kId,
          platform::errors::InvalidArgument(
              "The Variable type must be %s, but the type it holds is %s.",
              ToTypeName(VarTypeTrait<T>::kId), ToTypeName(holder_->Type())));
    }
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T>
  bool IsType() const {
    return holder_ && holder_->Type() == VarTypeTrait<T>::kId;
  }

  void Clear() { holder_.reset(); }

  int Type() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_, platform::errors::NotFound("Variable is not initialized."));
    return holder_->Type();
  }

  VariableInplaceVersion& InplaceVersionCounter() {
    return inplace_version_counter_;
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
        IsRegisteredVarType<T>(),
        "Not registered type. Please register T inside var_type_traits.h");
    PlaceholderImpl() { this->Init(&obj_, VarTypeTrait<T>::kId); }

   private:
    T obj_;
  };

  // pointers to a PlaceholderImpl object indeed.
  std::unique_ptr<Placeholder> holder_;
  VariableInplaceVersion inplace_version_counter_;
};

}  // namespace framework
}  // namespace paddle
