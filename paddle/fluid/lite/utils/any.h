// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <exception>
#include <memory>
#include <type_traits>
#include <typeinfo>

// This is an equivalent implementation of boost::any. We implement this to
// avoid including the whole boost library and keep the inference library small.
// These code references https://gist.github.com/shoooe/9202235

namespace paddle {
namespace lite {

class any;
template <class Type>
Type any_cast(any&);
template <class Type>
Type any_cast(const any&);
template <class Type>
Type* any_cast(any*);
template <class Type>
const Type* any_cast(const any*);
struct bad_any_cast : public std::bad_cast {};

class any {
 public:
  template <class Type>
  friend Type any_cast(any&);

  template <class Type>
  friend Type any_cast(const any&);

  template <class Type>
  friend Type* any_cast(any*);

  template <class Type>
  friend const Type* any_cast(const any*);

  any() : ptr(nullptr) {}
  explicit any(any&& x) : ptr(std::move(x.ptr)) {}

  explicit any(const any& x) {
    if (x.ptr) ptr = x.ptr->clone();
  }

  template <class Type>
  explicit any(const Type& x)
      : ptr(new concrete<typename std::decay<const Type>::type>(x)) {}
  any& operator=(any&& rhs) {
    ptr = std::move(rhs.ptr);
    return (*this);
  }
  any& operator=(const any& rhs) {
    ptr = std::move(any(rhs).ptr);
    return (*this);
  }
  template <class T>
  any& operator=(T&& x) {
    ptr.reset(new concrete<typename std::decay<T>::type>(
        typename std::decay<T>::type(x)));
    return (*this);
  }
  template <class T>
  any& operator=(const T& x) {
    ptr.reset(new concrete<typename std::decay<T>::type>(
        typename std::decay<T>::type(x)));
    return (*this);
  }
  void clear() { ptr.reset(nullptr); }
  bool empty() const { return ptr == nullptr; }
  const std::type_info& type() const {
    return (!empty()) ? ptr->type() : typeid(void);
  }

 private:
  struct placeholder {
    virtual std::unique_ptr<placeholder> clone() const = 0;
    virtual const std::type_info& type() const = 0;
    virtual ~placeholder() {}
  };

  template <class T>
  struct concrete : public placeholder {
    explicit concrete(T&& x) : value(std::move(x)) {}
    explicit concrete(const T& x) : value(x) {}
    virtual std::unique_ptr<placeholder> clone() const override {
      return std::unique_ptr<placeholder>(new concrete<T>(value));
    }
    virtual const std::type_info& type() const override { return typeid(T); }
    T value;
  };

  std::unique_ptr<placeholder> ptr;
};

template <class Type>
Type any_cast(any& val) {
  if (val.ptr->type() != typeid(Type)) throw bad_any_cast();
  return static_cast<any::concrete<Type>*>(val.ptr.get())->value;
}
template <class Type>
Type any_cast(const any& val) {
  return any_cast<Type>(any(val));
}
template <class Type>
Type* any_cast(any* ptr) {
  return dynamic_cast<Type*>(ptr->ptr.get());
}
template <class Type>
const Type* any_cast(const any* ptr) {
  return dynamic_cast<const Type*>(ptr->ptr.get());
}

}  // namespace lite
}  // namespace paddle
