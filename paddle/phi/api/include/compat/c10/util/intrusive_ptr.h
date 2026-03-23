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

// The file has been adapted from the PyTorch project.
// Licensed under BSD-style license:
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace c10 {

// Forward declarations
class intrusive_ptr_target;
namespace raw {
namespace intrusive_ptr {
inline void incref(intrusive_ptr_target* self);
inline void decref(intrusive_ptr_target* self);
}  // namespace intrusive_ptr
namespace weak_intrusive_ptr {
inline void incref(intrusive_ptr_target* self);
inline void decref(intrusive_ptr_target* self);
}  // namespace weak_intrusive_ptr
struct DontIncreaseRefcount {};
}  // namespace raw

namespace detail {

constexpr uint64_t kImpracticallyHugeReferenceCount = 0x0FFFFFFF;
constexpr uint64_t kImpracticallyHugeWeakReferenceCount =
    (kImpracticallyHugeReferenceCount << 32);
constexpr uint64_t kReferenceCountOne = 1;
constexpr uint64_t kWeakReferenceCountOne = (kReferenceCountOne << 32);
constexpr uint64_t kUniqueRef = (kReferenceCountOne | kWeakReferenceCountOne);

inline uint32_t refcount(uint64_t combined_refcount) {
  return static_cast<uint32_t>(combined_refcount);
}

inline uint32_t weakcount(uint64_t combined_refcount) {
  // Bit 63 is reserved for kHasPyObject in PyTorch (a flag indicating a live
  // Python wrapper). This compat layer does not implement the PyObject path,
  // so the bit will never be set, but we mask it out here to match PyTorch's
  // extraction logic and remain numerically correct if the bit were ever set.
  return static_cast<uint32_t>((combined_refcount & ~(uint64_t(1) << 63)) >>
                               32);
}

inline uint64_t atomic_combined_refcount_increment(
    std::atomic<uint64_t>* combined_refcount, uint64_t inc) {
  return combined_refcount->fetch_add(inc, std::memory_order_relaxed) + inc;
}

inline uint64_t atomic_combined_refcount_decrement(
    std::atomic<uint64_t>* combined_refcount, uint64_t dec) {
  return combined_refcount->fetch_sub(dec, std::memory_order_acq_rel) - dec;
}

inline uint32_t atomic_weakcount_increment(
    std::atomic<uint64_t>* combined_refcount) {
  return weakcount(atomic_combined_refcount_increment(combined_refcount,
                                                      kWeakReferenceCountOne));
}

inline uint32_t atomic_weakcount_decrement(
    std::atomic<uint64_t>* combined_refcount) {
  return weakcount(atomic_combined_refcount_decrement(combined_refcount,
                                                      kWeakReferenceCountOne));
}

template <class T>
struct intrusive_target_default_null_type final {
  static constexpr T* singleton() noexcept { return nullptr; }
};

}  // namespace detail

class intrusive_ptr_target {
 public:
  intrusive_ptr_target() noexcept : combined_refcount_(0) {}

  intrusive_ptr_target(intrusive_ptr_target&& /*other*/) noexcept
      : intrusive_ptr_target() {}

  intrusive_ptr_target& operator=(intrusive_ptr_target&& /*other*/) noexcept {
    return *this;
  }

  intrusive_ptr_target(const intrusive_ptr_target& /*other*/) noexcept
      : intrusive_ptr_target() {}

  intrusive_ptr_target& operator=(
      const intrusive_ptr_target& /*other*/) noexcept {
    return *this;
  }

  uint32_t refcount() const {
    return detail::refcount(combined_refcount_.load(std::memory_order_relaxed));
  }

  uint32_t weakcount() const {
    return detail::weakcount(
        combined_refcount_.load(std::memory_order_relaxed));
  }

 protected:
  virtual ~intrusive_ptr_target() = default;

 private:
  mutable std::atomic<uint64_t> combined_refcount_;

  template <typename T, typename NullType>
  friend class intrusive_ptr;
  template <typename T, typename NullType>
  friend class weak_intrusive_ptr;
  friend inline void raw::intrusive_ptr::incref(intrusive_ptr_target* self);
  friend inline void raw::intrusive_ptr::decref(intrusive_ptr_target* self);
  friend inline void raw::weak_intrusive_ptr::incref(
      intrusive_ptr_target* self);
  friend inline void raw::weak_intrusive_ptr::decref(
      intrusive_ptr_target* self);
};

namespace raw {
namespace intrusive_ptr {
inline void incref(intrusive_ptr_target* self) {
  if (self) {
    detail::atomic_combined_refcount_increment(&self->combined_refcount_,
                                               detail::kReferenceCountOne);
  }
}
inline void decref(intrusive_ptr_target* self) {
  if (self) {
    uint64_t new_count = detail::atomic_combined_refcount_decrement(
        &self->combined_refcount_, detail::kReferenceCountOne);
    if (detail::refcount(new_count) == 0) {
      // All strong references gone; release the implicit weak reference
      // (strong refs count as +1 to weakcount per the kUniqueRef invariant).
      if (detail::atomic_weakcount_decrement(&self->combined_refcount_) == 0) {
        delete self;
      }
    }
  }
}
}  // namespace intrusive_ptr
namespace weak_intrusive_ptr {
inline void incref(intrusive_ptr_target* self) {
  if (self) {
    detail::atomic_weakcount_increment(&self->combined_refcount_);
  }
}
inline void decref(intrusive_ptr_target* self) {
  if (self) {
    if (detail::atomic_weakcount_decrement(&self->combined_refcount_) == 0) {
      delete self;
    }
  }
}
}  // namespace weak_intrusive_ptr
}  // namespace raw

template <class TTarget, class NullType>
class weak_intrusive_ptr;

template <class TTarget,
          class NullType = detail::intrusive_target_default_null_type<TTarget>>
class intrusive_ptr final {
 private:
  static_assert(
      std::is_base_of_v<TTarget,
                        std::remove_pointer_t<decltype(NullType::singleton())>>,
      "NullType::singleton() must return a element_type* pointer");

  TTarget* target_;

  template <class TTarget2, class NullType2>
  friend class intrusive_ptr;
  friend class weak_intrusive_ptr<TTarget, NullType>;

  void retain_() noexcept {
    if (target_ != NullType::singleton()) {
      detail::atomic_combined_refcount_increment(&target_->combined_refcount_,
                                                 detail::kReferenceCountOne);
    }
  }

  void reset_() noexcept {
    if (target_ != NullType::singleton()) {
      uint64_t new_count = detail::atomic_combined_refcount_decrement(
          &target_->combined_refcount_, detail::kReferenceCountOne);
      if (detail::refcount(new_count) == 0) {
        // All strong references gone; release the implicit weak reference
        // (strong refs count as +1 to weakcount per the kUniqueRef invariant).
        if (detail::atomic_weakcount_decrement(&target_->combined_refcount_) ==
            0) {
          delete target_;
        }
      }
      target_ = NullType::singleton();
    }
  }

 public:
  using element_type = TTarget;
  using pointer = TTarget*;

  intrusive_ptr() noexcept : target_(NullType::singleton()) {}

  intrusive_ptr(std::nullptr_t) noexcept : target_(NullType::singleton()) {}

  explicit intrusive_ptr(TTarget* raw) : target_(raw) {
    if (target_ != NullType::singleton()) {
      target_->combined_refcount_.store(detail::kUniqueRef,
                                        std::memory_order_relaxed);
    }
  }

  intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) { retain_(); }

  intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  template <typename From, typename FromNullType>
  /* implicit */ intrusive_ptr(
      const intrusive_ptr<From, FromNullType>& rhs) noexcept
      : target_(rhs.target_) {
    static_assert(std::is_convertible_v<From*, TTarget*>,
                  "Source type must be convertible to target type");
    retain_();
  }

  template <typename From, typename FromNullType>
  /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(rhs.target_) {
    static_assert(std::is_convertible_v<From*, TTarget*>,
                  "Source type must be convertible to target type");
    rhs.target_ = FromNullType::singleton();
  }

  ~intrusive_ptr() { reset_(); }

  intrusive_ptr& operator=(const intrusive_ptr& rhs) {
    if (this != &rhs) {
      reset_();
      target_ = rhs.target_;
      retain_();
    }
    return *this;
  }

  intrusive_ptr& operator=(intrusive_ptr&& rhs) noexcept {
    if (this != &rhs) {
      reset_();
      target_ = rhs.target_;
      rhs.target_ = NullType::singleton();
    }
    return *this;
  }

  // Takes ownership of a raw pointer without incrementing the refcount.
  static intrusive_ptr reclaim(TTarget* raw_ptr) {
    intrusive_ptr result;
    result.target_ = raw_ptr;
    return result;
  }

  // unsafe_adopt is a PyTorch API compatibility alias for reclaim().
  // Both adopt a raw pointer without incrementing the refcount; prefer
  // reclaim() in new code.
  static intrusive_ptr unsafe_adopt(TTarget* raw_ptr) {
    return reclaim(raw_ptr);
  }

  TTarget* get() const noexcept { return target_; }

  TTarget& operator*() const { return *target_; }

  TTarget* operator->() const { return target_; }

  explicit operator bool() const noexcept {
    return target_ != NullType::singleton();
  }

  uint32_t use_count() const noexcept {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->refcount();
  }

  bool defined() const noexcept { return target_ != NullType::singleton(); }

  bool unique() const noexcept { return use_count() == 1; }

  void reset() noexcept { reset_(); }

  void swap(intrusive_ptr& other) noexcept {
    using std::swap;
    swap(target_, other.target_);
  }

  [[deprecated(
      "intrusive_ptr::release is unsafe; use reclaim() or explicit ownership "
      "transfer instead")]] TTarget*
  release() noexcept {
    TTarget* result = target_;
    target_ = NullType::singleton();
    return result;
  }

  bool operator==(const intrusive_ptr& rhs) const noexcept {
    return target_ == rhs.target_;
  }
  bool operator!=(const intrusive_ptr& rhs) const noexcept {
    return target_ != rhs.target_;
  }
  bool operator==(std::nullptr_t) const noexcept {
    return target_ == NullType::singleton();
  }
  bool operator!=(std::nullptr_t) const noexcept {
    return target_ != NullType::singleton();
  }
};

template <class TTarget,
          class NullType = detail::intrusive_target_default_null_type<TTarget>>
class weak_intrusive_ptr final {
 private:
  TTarget* target_;

  template <class TTarget2, class NullType2>
  friend class weak_intrusive_ptr;
  friend class intrusive_ptr<TTarget, NullType>;

  void retain_() {
    if (target_ != NullType::singleton()) {
      detail::atomic_weakcount_increment(&target_->combined_refcount_);
    }
  }

  void reset_() noexcept {
    if (target_ != NullType::singleton()) {
      if (detail::atomic_weakcount_decrement(&target_->combined_refcount_) ==
          0) {
        delete target_;
      }
      target_ = NullType::singleton();
    }
  }

 public:
  using element_type = TTarget;

  weak_intrusive_ptr() noexcept : target_(NullType::singleton()) {}

  weak_intrusive_ptr(const intrusive_ptr<TTarget, NullType>& p)
      : target_(p.target_) {
    retain_();
  }

  weak_intrusive_ptr(const weak_intrusive_ptr& rhs) : target_(rhs.target_) {
    retain_();
  }

  weak_intrusive_ptr(weak_intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
    rhs.target_ = NullType::singleton();
  }

  ~weak_intrusive_ptr() { reset_(); }

  weak_intrusive_ptr& operator=(const weak_intrusive_ptr& rhs) {
    if (this != &rhs) {
      reset_();
      target_ = rhs.target_;
      retain_();
    }
    return *this;
  }

  weak_intrusive_ptr& operator=(weak_intrusive_ptr&& rhs) {
    if (this != &rhs) {
      reset_();
      target_ = rhs.target_;
      rhs.target_ = NullType::singleton();
    }
    return *this;
  }

  intrusive_ptr<TTarget, NullType> lock() const {
    if (target_ == NullType::singleton()) {
      return intrusive_ptr<TTarget, NullType>();
    }
    auto& atomic = target_->combined_refcount_;
    uint64_t count = atomic.load(std::memory_order_relaxed);
    while (true) {
      if (detail::refcount(count) == 0) {
        return intrusive_ptr<TTarget, NullType>();
      }
      if (atomic.compare_exchange_weak(count,
                                       count + detail::kReferenceCountOne,
                                       std::memory_order_acq_rel,
                                       std::memory_order_relaxed)) {
        return intrusive_ptr<TTarget, NullType>::unsafe_adopt(target_);
      }
    }
  }

  uint32_t use_count() const {
    if (target_ == NullType::singleton()) {
      return 0;
    }
    return target_->refcount();
  }

  bool expired() const { return use_count() == 0; }

  void reset() { reset_(); }
};

// Creates a new T with an initial strong refcount of 1.
template <typename T, typename... Args>
intrusive_ptr<T> make_intrusive(Args&&... args) {
  return intrusive_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace c10
