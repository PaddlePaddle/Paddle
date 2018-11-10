/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {

/**
 * \brief   Allocate memory block in one place.
 *
 * \param[in]  place  Allocation place (CPU or GPU).
 * \param[in]  size   Allocation size.
 *
 * \return  Allocated memory block address.
 *
 * \note    If return nullptr, it indicates memory allocation failed
 *          because insufficient memory in current system. When Alloc
 *          function is invoked, you must check the returned memory
 *          address is valid or not.
 */
template <typename Place>
void* Alloc(Place place, size_t size);

/**
 * \brief   Free memory block in one place.
 *
 * \param[in]  place  Allocation place (CPU or GPU).
 * \param[in]  ptr    Memory block address to free.
 *
 */
template <typename Place>
void Free(Place place, void* ptr);

/**
 * \brief   Total size of used memory in one place.
 *
 * \param[in]  place  Allocation place (CPU or GPU).
 *
 */
template <typename Place>
size_t Used(Place place);

struct Usage : public boost::static_visitor<size_t> {
  size_t operator()(const platform::CPUPlace& cpu) const;
  size_t operator()(const platform::CUDAPlace& gpu) const;
  size_t operator()(const platform::CUDAPinnedPlace& cuda_pinned) const;
};

size_t memory_usage(const platform::Place& p);

/**
 * \brief   Free memory block in one place.
 *
 * \note    In some cases, custom deleter is used to
 *          deallocate the memory automatically for
 *          std::unique_ptr<T> in tensor.h.
 *
 */
template <typename T, typename Place>
class PODDeleter {
  static_assert(std::is_pod<T>::value, "T must be POD");

 public:
  explicit PODDeleter(Place place) : place_(place) {}
  void operator()(T* ptr) { Free(place_, static_cast<void*>(ptr)); }

 private:
  Place place_;
};

/**
 * \brief   Free memory block in one place does not meet POD
 *
 * \note    In some cases, custom deleter is used to
 *          deallocate the memory automatically for
 *          std::unique_ptr<T> in tensor.h.
 *
 */
template <typename T, typename Place>
class PlainDeleter {
 public:
  explicit PlainDeleter(Place place) : place_(place) {}
  void operator()(T* ptr) { Free(place_, reinterpret_cast<void*>(ptr)); }

 private:
  Place place_;
};

}  // namespace memory
}  // namespace paddle
