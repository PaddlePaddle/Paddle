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

#include "paddle/platform/assert.h"

namespace paddle {
namespace framework {

class Variable {
 public:
  template <typename T>
  const T& Get() const {
    PADDLE_ASSERT(IsType<T>());
    return *static_cast<const T*>(holder_->Ptr());
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
};

}  // namespace framework
}  // namespace paddle
