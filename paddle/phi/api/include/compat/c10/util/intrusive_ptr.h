// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <memory>
#include <utility>

namespace c10 {

/**
 * intrusive_ptr<T> — a reference-counted smart pointer compatible with
 * PyTorch's c10::intrusive_ptr interface, backed by std::shared_ptr.
 *
 * This is intentionally NOT a true intrusive refcount implementation;
 * it simply wraps std::shared_ptr so that existing Paddle code that
 * uses shared_ptr-based ownership "just works", while preserving the
 * PyTorch-facing API surface.
 */
template <typename T>
class intrusive_ptr {
 public:
  using element_type = T;
  using pointer = T*;

  intrusive_ptr() : ptr_(nullptr) {}

  explicit intrusive_ptr(T* raw) : ptr_(std::shared_ptr<T>(raw)) {}

  /* implicit */ intrusive_ptr(std::shared_ptr<T> ptr)  // NOLINT
      : ptr_(std::move(ptr)) {}

  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
  /* implicit */ intrusive_ptr(const intrusive_ptr<U>& other)  // NOLINT
      : ptr_(other.get_shared()) {}

  template <typename... Args>
  static intrusive_ptr<T> make(Args&&... args) {
    return intrusive_ptr<T>(std::make_shared<T>(std::forward<Args>(args)...));
  }

  // ---- observers -----------------------------------------------------------
  T* get() const noexcept { return ptr_.get(); }
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_.get(); }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  uint32_t use_count() const noexcept { return ptr_.use_count(); }

  bool defined() const noexcept { return ptr_ != nullptr; }

  // Access the underlying shared_ptr (needed by some interop layers).
  const std::shared_ptr<T>& get_shared() const noexcept { return ptr_; }

  // ---- mutators ------------------------------------------------------------

  /// Releases ownership and returns the raw pointer.
  /// After this call, the intrusive_ptr is empty.
  /// NOTE: Unlike unique_ptr::release(), the reference count is NOT
  /// decremented — the caller is responsible for the lifetime.
  T* release() noexcept {
    T* raw = ptr_.get();
    ptr_.reset();
    return raw;
  }

  void reset() noexcept { ptr_.reset(); }

  // ---- comparison ----------------------------------------------------------
  bool operator==(const intrusive_ptr& rhs) const noexcept {
    return ptr_ == rhs.ptr_;
  }
  bool operator!=(const intrusive_ptr& rhs) const noexcept {
    return ptr_ != rhs.ptr_;
  }

 private:
  std::shared_ptr<T> ptr_;
};

/// Factory function mirroring c10::make_intrusive<T>(args...).
template <typename T, typename... Args>
intrusive_ptr<T> make_intrusive(Args&&... args) {
  return intrusive_ptr<T>::make(std::forward<Args>(args)...);
}

}  // namespace c10
