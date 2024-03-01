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

#include <atomic>

namespace phi {

template <typename DerivedT>
class intrusive_ref_counter;
template <typename DerivedT>
void intrusive_ptr_add_ref(const intrusive_ref_counter<DerivedT>* p) noexcept;
template <typename DerivedT>
void intrusive_ptr_release(const intrusive_ref_counter<DerivedT>* p) noexcept;

template <typename DerivedT>
class intrusive_ref_counter {
 public:
  constexpr intrusive_ref_counter() noexcept : ref_(1) {}
  virtual ~intrusive_ref_counter() = default;

  unsigned int use_count() const noexcept { return ref_.load(); }

 protected:
  intrusive_ref_counter(const intrusive_ref_counter&) = delete;
  intrusive_ref_counter& operator=(const intrusive_ref_counter&) = delete;

  friend void intrusive_ptr_add_ref<DerivedT>(
      const intrusive_ref_counter<DerivedT>* p) noexcept;
  friend void intrusive_ptr_release<DerivedT>(
      const intrusive_ref_counter<DerivedT>* p) noexcept;

 private:
  mutable std::atomic_int_fast32_t ref_;
};

template <typename DerivedT>
inline void intrusive_ptr_add_ref(
    const intrusive_ref_counter<DerivedT>* p) noexcept {
  p->ref_.fetch_add(1, std::memory_order_relaxed);
}

template <typename DerivedT>
inline void intrusive_ptr_release(
    const intrusive_ref_counter<DerivedT>* p) noexcept {
  if (p->ref_.load(std::memory_order_acquire) == 0 ||
      p->ref_.fetch_sub(1) == 0) {
    delete static_cast<const DerivedT*>(p);  // NOLINT
  }
}

}  // namespace phi
