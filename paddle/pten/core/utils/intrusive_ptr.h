/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <utility>
#include "glog/logging.h"
#include "paddle/pten/core/enforce.h"

namespace pten {

template <typename T>
class intrusive_ptr {
 public:
  using this_type = intrusive_ptr;
  constexpr intrusive_ptr() noexcept = default;

  ~intrusive_ptr() {
    if (px) {
      intrusive_ptr_release(px);
    }
  }

  intrusive_ptr(intrusive_ptr&& rhs) noexcept : px(rhs.px) { rhs.px = nullptr; }

  template <typename U,
            typename = std::enable_if_t<std::is_base_of<T, U>::value>>
  intrusive_ptr(intrusive_ptr<U>&& rhs) noexcept : px(rhs.get()) {
    rhs.reset();
  }

  intrusive_ptr& operator=(intrusive_ptr&& rhs) {
    swap(rhs);
    return *this;
  }

  void reset() { this_type().swap(*this); }

  void reset(T* rhs) { this_type(rhs).swap(*this); }

  void reset(T* rhs, bool add_ref) { this_type(rhs, add_ref).swap(*this); }

  T* get() const noexcept { return px; }

  T* detach() noexcept {
    T* ret = px;
    px = nullptr;
    return ret;
  }

  T& operator*() const {
    PADDLE_ENFORCE_NOT_NULL(
        px,
        pten::errors::PreconditionNotMet(
            "The pointer must be non-null before the dereference operation."));
    return *px;
  }

  T* operator->() const {
    PADDLE_ENFORCE_NOT_NULL(
        px,
        pten::errors::PreconditionNotMet(
            "The pointer must be non-null before the dereference operation."));
    return px;
  }

  void swap(intrusive_ptr& rhs) noexcept {
    T* tmp = px;
    px = rhs.px;
    rhs.px = tmp;
  }

 private:
  template <typename U,
            typename = std::enable_if_t<std::is_base_of<T, U>::value>>
  explicit intrusive_ptr(U* p, bool add_ref = true) : px(p) {
    if (px && add_ref) {
      intrusive_ptr_add_ref(px);
    }
  }

  template <typename R, typename... Args>
  friend intrusive_ptr<R> make_intrusive(Args&&...);
  template <typename R>
  friend intrusive_ptr<R> copy_intrusive(const intrusive_ptr<R>&);

  T* px{nullptr};
};

template <typename T, typename U>
inline bool operator==(const intrusive_ptr<T>& a,
                       const intrusive_ptr<U>& b) noexcept {
  return a.get() == b.get();
}

template <typename T, typename U>
inline bool operator!=(const intrusive_ptr<T>& a,
                       const intrusive_ptr<U>& b) noexcept {
  return a.get() != b.get();
}

template <typename T, typename U>
inline bool operator==(const intrusive_ptr<T>& a, U* b) noexcept {
  return a.get() == b;
}

template <typename T, typename U>
inline bool operator!=(const intrusive_ptr<T>& a, U* b) noexcept {
  return a.get() != b;
}

template <typename T, typename U>
inline bool operator==(T* a, const intrusive_ptr<U>& b) noexcept {
  return a == b.get();
}

template <typename T, typename U>
inline bool operator!=(T* a, const intrusive_ptr<U>& b) noexcept {
  return a != b.get();
}

template <typename T>
inline bool operator==(const intrusive_ptr<T>& p, std::nullptr_t) noexcept {
  return p.get() == nullptr;
}

template <typename T>
inline bool operator==(std::nullptr_t, const intrusive_ptr<T>& p) noexcept {
  return p.get() == nullptr;
}

template <typename T>
inline bool operator!=(const intrusive_ptr<T>& p, std::nullptr_t) noexcept {
  return p.get() != nullptr;
}

template <typename T>
inline bool operator!=(std::nullptr_t, const intrusive_ptr<T>& p) noexcept {
  return p.get() != nullptr;
}

template <typename T, typename... Args>
inline intrusive_ptr<T> make_intrusive(Args&&... args) {
  return intrusive_ptr<T>(new T(std::forward<Args>(args)...), false);
}

template <typename T>
inline intrusive_ptr<T> copy_intrusive(const intrusive_ptr<T>& rhs) {
  return intrusive_ptr<T>(rhs.get(), true);
}

}  // namespace pten
