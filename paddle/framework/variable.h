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

#include <typeinfo>

namespace paddle {
namespace framework {

class Variable {
 public:
  template <typename T>
  const T& Get() const {
    return *static_cast<const T*>(holder_->Ptr());
  }

  template <typename T>
  T* GetMutable() {
    if (holder_ != nullptr && typeid(T) == holder_->Type()) {
      return static_cast<T*>(holder_->Ptr());
    } else {
      return Reset<T>(new T(), DefaultDeleter<T>());
    }
  }

  ~Variable() {
    if (holder_ != nullptr) delete holder_;
  }

 private:
  // DefaultDeleter is functor which uses C++'s delete(T*).
  template <typename T>
  struct DefaultDeleter {
    void operator()(T* ptr) { delete ptr; }
  };

  struct Placeholder {
    virtual ~Placeholder() {}
    virtual const std::type_info& Type() const = 0;
    virtual void* Ptr() const = 0;
  };

  // Placeholder hides type T, so it doesn't appear as a template
  // parameter of Variable.
  template <typename T>
  struct PlaceholderImpl : public Placeholder {
    typedef std::function<void(T*)> Deleter;

    PlaceholderImpl(T* ptr) : ptr_(ptr), type_(typeid(T)) {}
    PlaceholderImpl(T* ptr, Deleter d)
        : ptr_(ptr), type_(typeid(T)), deleter_(d) {}

    virtual ~PlaceholderImpl() {
      deleter_(ptr_);
      ptr_ = nullptr;
    }
    virtual const std::type_info& Type() const { return type_; }
    virtual void* Ptr() const { return ptr_; }

    T* ptr_ = nullptr;
    const std::type_info& type_;
    std::function<void(T*)> deleter_ = DefaultDeleter<T>();
  };

  template <typename T>
  T* Reset(T* allocated, typename PlaceholderImpl<T>::Deleter deleter) {
    if (holder_ != nullptr) {
      delete holder_;
    }
    holder_ = new PlaceholderImpl<T>(allocated, deleter);
    return allocated;
  }

  Placeholder* holder_;  // pointers to a PlaceholderImpl object indeed.
};

}  // namespace framework
}  // namespace paddle
