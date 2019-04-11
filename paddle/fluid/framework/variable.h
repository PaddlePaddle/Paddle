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

class Variable {
 public:
  template <typename T>
  const T& Get() const {
    static_assert(
        IsRegisteredVarType<T>(),
        "Not registered type. Please register T inside var_type_traits.h");
    PADDLE_ENFORCE(holder_ != nullptr, "Variable must hold some thing");
    PADDLE_ENFORCE(holder_->Type() == VarTypeTrait<T>::kId,
                   "Variable must be type %s, the holding type is %s",
                   ToTypeName(VarTypeTrait<T>::kId),
                   ToTypeName(holder_->Type()));
    return *static_cast<const T*>(holder_->Ptr());
  }

  bool IsInitialized() const { return holder_ != nullptr; }

  template <typename T>
  T* GetMutable() {
    if (!holder_) {
      holder_.reset(new PlaceholderImpl<T>());
    } else {
      PADDLE_ENFORCE(holder_->Type() == VarTypeTrait<T>::kId,
                     "Variable must be type %s, the holding type is %s",
                     ToTypeName(VarTypeTrait<T>::kId),
                     ToTypeName(holder_->Type()));
    }
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T>
  bool IsType() const {
    return holder_ && holder_->Type() == VarTypeTrait<T>::kId;
  }

  void Clear() { holder_.reset(); }

  int Type() const {
    PADDLE_ENFORCE(holder_ != nullptr, "Must hold memory");
    return holder_->Type();
  }

 private:
  struct Placeholder {
    virtual ~Placeholder() = default;

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
};

}  // namespace framework
}  // namespace paddle
