/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
#pragma once

#include <memory>
#include <typeindex>
#include <typeinfo>

#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

class Variable {
 public:
  template <typename T>
  const T& Get() const {
    PADDLE_ENFORCE(IsType<T>(), "Variable must be type %s", typeid(T).name());
    return *static_cast<const T*>(holder_->Ptr());
  }

  // Specialized template for Tensor,
  // so that the Tensor can be got from LoDTensor.
  // How to get Tensor from LoDTensor also can be put in InferShapeContext.
  template <>
  const Tensor& Get<Tensor>() const {
    if (IsType<LoDTensor>()) {
      auto lod_t = static_cast<const LoDTensor*>(holder_->Ptr());
      return *lod_t->tensor();
    } else {
      PADDLE_ENFORCE(IsType<Tensor>(),
                     "Variable must be type LoDTensor or Tensor");
      return *static_cast<const Tensor*>(holder_->Ptr());
    }
  }

  template <typename T>
  T* GetMutable() {
    if (!IsType<T>()) {
      holder_.reset(new PlaceholderImpl<T>(new T()));
    }
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T>
  bool IsType() const {
    return holder_ != nullptr &&
           std::type_index(typeid(T)) == std::type_index(holder_->Type());
  }

 private:
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual const std::type_info& Type() const = 0;
    virtual void* Ptr() const = 0;
  };

  // Placeholder hides type T, so it doesn't appear as a template
  // parameter of Variable.
  template <typename T>
  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(T* ptr) : ptr_(ptr), type_(typeid(T)) {}

    virtual const std::type_info& Type() const { return type_; }
    virtual void* Ptr() const { return static_cast<void*>(ptr_.get()); }

    std::unique_ptr<T> ptr_;
    const std::type_info& type_;
  };

  std::unique_ptr<Placeholder>
      holder_;  // pointers to a PlaceholderImpl object indeed.

  // name_ is only meaningful with a Scope and accessible by it.
  //
  // NOTE: Please don't expose name_ by adding methods like
  // Variable::Name or Scope::VarName!  A variable could have a human
  // readable name or an auto-generated scope-unique name.  In the
  // former case, the caller knows the name and doesn't need to access
  // the name; in the latter case, the variable should be identified
  // by its address but not the unreadable name.
  friend class Scope;
  const std::string* name_;
};

}  // namespace framework
}  // namespace paddle
