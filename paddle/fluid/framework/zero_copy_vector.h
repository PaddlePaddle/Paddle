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

#include <initializer_list>
#include <vector>
#include "paddle/fluid/memory/malloc.h"

namespace paddle {
namespace framework {

template <typename T>
class ZeroCopyVector {
 public:
  using value_type = T;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  // Create an empty vector
  ZeroCopyVector() : allocation_(nullptr) {}

  // Create a vector initlizeded with a specified list
  ZeroCopyVector(std::initializer_list<T> list) {
    T *ptr = MutableData(list.size());
    int i = 0;
    for (T v : list) {
      ptr[i++] = v;
    }
  }

  explicit ZeroCopyVector(const std::vector<T> &other) {
    T *ptr = MutableData(other.size());
    int i = 0;
    for (T v : other) {
      ptr[i++] = v;
    }
  }

  // Access CPU data. Mutable.
  T &operator[](size_t i) { return MutableData()[i]; }

  // Access CPU data. Immutable.
  const T &operator[](size_t i) const { return data()[i]; }

  size_t size() const { return size_; }

  T &front() { return MutableData()[0]; }

  T &back() { return MutableData()[size_ - 1]; }

  const T &front() const { return data()[0]; }

  const T &back() const { return data()[size_ - 1]; }

  T *data() { return MutableData(); }

  const T *data() const { return reinterpret_cast<T *>(allocation_->ptr()); }

  T *MutableData() { return MutableData(size_); }

  // Assign from iterator.
  // template <typename Iter>
  // void assign(Iter begin, Iter end) {
  //   MutableData();
  //   int i = 0;
  //   for (Iter& iter = begin; iter < end; iter++) {
  //     allocation_[i++] = ;
  //   }
  // }

  void push_back(T elem) {
    size_++;
    MutableData()[size_ - 1] = elem;
  }

  void resize(size_t size) { size_ = size; }

  // Get CUDA pointer. Immutable.
  const T *CUDAData(platform::Place place) const {
    void *d_ptr = nullptr;
    cudaHostGetDevicePointer(&d_ptr, allocation_->ptr(), 0);
    return reinterpret_cast<T *>(d_ptr);
  }

  // Get CUDA pointer. Mutable.
  T *CUDAMutableData(platform::Place place) {
    void *d_ptr = nullptr;
    cudaHostGetDevicePointer(&d_ptr, allocation_->ptr(), 0);
    return reinterpret_cast<T *>(d_ptr);
  }

  void clear() { size_ = 0; }

  size_t capacity() const { return allocation_->size(); }

  const T *Data(platform::Place place) {
    if (platform::is_gpu_place(place)) {
      return CUDAData(place);
    } else {
      return data();
    }
  }

  T *MutableData(platform::Place place) {
    if (platform::is_gpu_place(place)) {
      return CUDAMutableData(place);
    } else {
      return data();
    }
  }

 private:
  T *MutableData(size_t size) {
    if (!allocation_ || allocation_->size() < size) {
      size_ = size;
      allocation_ = memory::Alloc(place_, size_ * sizeof(T));
    }
    return reinterpret_cast<T *>(allocation_->ptr());
  }

#ifdef PADDLE_WITH_CUDA
  size_t size_;  // size_ is the actually used space in allocation_.
  platform::CUDAPinnedPlace place_;
  mutable paddle::memory::AllocationPtr allocation_;
#endif
};

}  // namespace framework
}  // namespace paddle
