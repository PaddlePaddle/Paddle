// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

struct CacheAllocator {
  typedef char value_type;
  explicit CacheAllocator(phi::Place place) {
    VLOG(2) << "construct allocator";
    place_ = place;
  }

  ~CacheAllocator() { VLOG(2) << "destroy allocator"; }

  char* allocate(std::ptrdiff_t num_bytes) {
    VLOG(2) << "allocate " << num_bytes << " bytes";
    auto storage = memory_utils::AllocShared(place_, num_bytes);
    char* ptr = reinterpret_cast<char*>(storage->ptr());
    busy_allocation_.emplace(std::make_pair(ptr, storage));
    return ptr;
  }

  void deallocate(char* ptr, size_t) {
    VLOG(2) << "deallocate ";
    allocation_map_type::iterator iter = busy_allocation_.find(ptr);
    CHECK(iter != busy_allocation_.end());
    PADDLE_ENFORCE_NE(iter,
                      busy_allocation_.end(),
                      common::errors::InvalidArgument(
                          "Deallocate failed, can not find right position"));
    busy_allocation_.erase(iter);
  }

 private:
  typedef std::unordered_map<char*, std::shared_ptr<phi::Allocation>>
      allocation_map_type;
  allocation_map_type busy_allocation_;
  phi::Place place_;
};

template <typename T, bool kIsForward>
struct ReorderFunctor {
  ReorderFunctor(const T* x, const int64_t* shuffle_idx, T* y, int64_t stride)
      : x_(x), shuffle_idx_(shuffle_idx), y_(y), stride_(stride) {}

  HOSTDEVICE void operator()(int64_t idx) {
    auto reorder_idx = shuffle_idx_[idx / stride_] * stride_ + idx % stride_;
    if (kIsForward) {
      y_[idx] = x_[reorder_idx];
    } else {
      y_[reorder_idx] = x_[idx];
    }
  }

 private:
  const T* x_;
  const int64_t* shuffle_idx_;
  T* y_;
  int64_t stride_;
};

}  // namespace phi
#endif
