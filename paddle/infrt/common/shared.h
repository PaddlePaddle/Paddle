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
#include <atomic>
#include <string>
#include <type_traits>

namespace infrt {
namespace common {

class RefCount {
 public:
  using value_type = int32_t;
  RefCount() = default;

  value_type Inc() { return ++count_; }
  value_type Dec() { return --count_; }
  bool is_zero() const { return 0 == count_; }
  std::string to_string() { return std::to_string(count_.load()); }
  int32_t val() const { return count_; }

 private:
  std::atomic<value_type> count_{0};
};

class Object;
/**
 * The templated methods are used to unify the way to get the RefCount instance
 * in client classes.
 */
template <typename T>
RefCount& ref_count(const T* t) {
  static_assert(std::is_base_of<Object, T>::value, "T is not a Object");
  return t->__ref_count__;
}
template <typename T>
void Destroy(const T* t) {
  delete t;
}

template <typename T>
struct Shared {
  using object_ptr = T*;

  Shared() = default;
  explicit Shared(T* p) : p_(p) {
    if (p) IncRef(p);
  }
  Shared(const Shared& other) : p_(other.p_) { IncRef(p_); }
  Shared(Shared&& other) : p_(other.p_) { other.p_ = nullptr; }
  Shared<T>& operator=(const Shared<T>& other);

  //! Reset to another pointer \p x.
  void Reset(T* x = nullptr);

  //! Access the pointer in various ways.
  // @{
  inline T* get() const { return p_; }
  inline T& operator*() const { return *p_; }
  inline T* operator->() const { return p_; }
  inline T* self() { return p_; }
  inline const T* self() const { return p_; }
  // @}

  inline bool same_as(const Shared& other) { return p_ == other.p_; }
  inline bool defined() const { return p_; }
  inline bool operator<(const Shared& other) const { return p_ < other.p_; }
  inline Shared<T>& operator=(T* x);
  inline bool operator==(const Shared& other) const { return p_ == other.p_; }

  ~Shared();

 private:
  //! Increase the share count.
  void IncRef(T* p);

  //! Decrease the share count.
  void DecRef(T* p);

 protected:
  T* p_{};
};

template <typename T>
void Shared<T>::IncRef(T* p) {
  if (p) {
    ref_count(p).Inc();
  }
}
template <typename T>
void Shared<T>::DecRef(T* p) {
  if (p) {
    if (ref_count(p).Dec() == 0) {
      Destroy(p);
    }
  }
}
template <typename T>
Shared<T>& Shared<T>::operator=(const Shared<T>& other) {
  if (other.p_ == p_) return *this;
  // Other can be inside of something owned by this, so we should be careful to
  // incref other before we decref
  // ourselves.
  T* tmp = other.p_;
  IncRef(tmp);
  DecRef(p_);
  p_ = tmp;
  return *this;
}

template <typename T, typename... Args>
T* make_shared(Args&&... args) {
  return new T(args...);
}

template <typename T>
Shared<T>& Shared<T>::operator=(T* x) {
  if (p_ == x) return *this;

  T* tmp = x;
  IncRef(tmp);
  DecRef(p_);
  p_ = tmp;
  return *this;
}

template <typename T>
Shared<T>::~Shared() {
  DecRef(p_);
  p_ = nullptr;
}

template <typename T>
void Shared<T>::Reset(T* x) {
  if (x) IncRef(x);
  DecRef(p_);
  p_ = x;
}

}  // namespace common
}  // namespace infrt
