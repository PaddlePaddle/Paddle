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

#include <algorithm>
#include <initializer_list>
#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"

#include "glog/logging.h"

namespace paddle {
namespace framework {

// Vector<T> implements the std::vector interface, and can get Data or
// MutableData from any place. The data will be synced implicitly inside.
template <typename T>
class Vector {
 public:
  using value_type = T;

  // Default ctor. Create empty Vector
  Vector() { InitEmpty(); }

  // Fill vector with value. The vector size is `count`.
  explicit Vector(size_t count, const T& value = T()) {
    InitEmpty();
    if (count != 0) {
      resize(count);
      T* ptr = begin();
      for (size_t i = 0; i < count; ++i) {
        ptr[i] = value;
      }
    }
  }

  // Ctor with init_list
  Vector(std::initializer_list<T> init) {
    if (init.size() == 0) {
      InitEmpty();
    } else {
      InitByIter(init.size(), init.begin(), init.end());
    }
  }

  // implicit cast from std::vector.
  template <typename U>
  Vector(const std::vector<U>& dat) {  // NOLINT
    if (dat.size() == 0) {
      InitEmpty();
    } else {
      InitByIter(dat.size(), dat.begin(), dat.end());
    }
  }

  // Copy ctor
  Vector(const Vector<T>& other) { this->operator=(other); }

  // Copy operator
  Vector<T>& operator=(const Vector<T>& other) {
    if (other.size() != 0) {
      this->InitByIter(other.size(), other.begin(), other.end());
    } else {
      InitEmpty();
    }
    return *this;
  }

  // Move ctor
  Vector(Vector<T>&& other) {
    this->size_ = other.size_;
    this->flag_ = other.flag_;
    if (other.cuda_vec_.memory_size()) {
      this->cuda_vec_.ShareDataWith(other.cuda_vec_);
    }
    if (other.cpu_vec_.memory_size()) {
      this->cpu_vec_.ShareDataWith(other.cpu_vec_);
    }
  }

  // CPU data access method. Mutable.
  T& operator[](size_t i) {
    MutableCPU();
    return const_cast<T*>(cpu_vec_.data<T>())[i];
  }

  // CPU data access method. Immutable.
  const T& operator[](size_t i) const {
    ImmutableCPU();
    return cpu_vec_.data<T>()[i];
  }

  // std::vector iterator methods. Based on CPU data access method
  size_t size() const { return size_; }

  T* begin() { return capacity() == 0 ? &EmptyDummy() : &this->operator[](0); }

  T* end() {
    return capacity() == 0 ? &EmptyDummy() : &this->operator[](size());
  }

  T& front() { return *begin(); }

  T& back() {
    auto it = end();
    --it;
    return *it;
  }

  const T* begin() const {
    return capacity() == 0 ? &EmptyDummy() : &this->operator[](0);
  }

  const T* end() const {
    return capacity() == 0 ? &EmptyDummy() : &this->operator[](size());
  }

  const T* cbegin() const { return begin(); }

  const T* cend() const { return end(); }

  const T& back() const {
    auto it = end();
    --it;
    return *it;
  }

  T* data() { return begin(); }

  const T* data() const { return begin(); }

  const T& front() const { return *begin(); }
  // end of std::vector iterator methods

  // assign this from iterator.
  // NOTE: the iterator must support `end-begin`
  template <typename Iter>
  void assign(Iter begin, Iter end) {
    InitByIter(end - begin, begin, end);
  }

  // push_back. If the previous capacity is not enough, the memory will
  // double.
  void push_back(T elem) {
    if (size_ + 1 > capacity()) {
      reserve((size_ + 1) << 1);
    }
    *end() = elem;
    ++size_;
  }

  // extend a vector by iterator.
  // NOTE: the iterator must support end-begin
  template <typename It>
  void Extend(It begin, It end) {
    size_t pre_size = size_;
    resize(pre_size + (end - begin));
    T* ptr = this->begin() + pre_size;
    for (; begin < end; ++begin, ++ptr) {
      *ptr = *begin;
    }
  }

  // resize the vector
  void resize(size_t size) {
    if (size + 1 <= capacity()) {
      size_ = size;
    } else {
      MutableCPU();
      Tensor cpu_tensor;
      platform::Place cpu = platform::CPUPlace();
      T* ptr = cpu_tensor.mutable_data<T>(
          framework::make_ddim({static_cast<int64_t>(size)}), cpu);
      const T* old_ptr =
          cpu_vec_.memory_size() == 0 ? nullptr : cpu_vec_.data<T>();
      if (old_ptr != nullptr) {
        std::copy(old_ptr, old_ptr + size_, ptr);
      }
      size_ = size;
      cpu_vec_.ShareDataWith(cpu_tensor);
    }
  }

  // get cuda ptr. immutable
  const T* CUDAData(platform::Place place) const {
    PADDLE_ENFORCE(platform::is_gpu_place(place),
                   "CUDA Data must on CUDA place");
    ImmutableCUDA(place);
    return cuda_vec_.data<T>();
  }

  // get cuda ptr. mutable
  T* CUDAMutableData(platform::Place place) {
    const T* ptr = CUDAData(place);
    flag_ = kDirty | kDataInCUDA;
    return const_cast<T*>(ptr);
  }

  // clear
  void clear() {
    size_ = 0;
    flag_ = kDirty | kDataInCPU;
  }

  size_t capacity() const {
    return cpu_vec_.memory_size() / SizeOfType(typeid(T));
  }

  // reserve data
  void reserve(size_t size) {
    size_t pre_size = size_;
    resize(size);
    resize(pre_size);
  }

  // the unify method to access CPU or CUDA data. immutable.
  const T* Data(platform::Place place) const {
    if (platform::is_gpu_place(place)) {
      return CUDAData(place);
    } else {
      return data();
    }
  }

  // the unify method to access CPU or CUDA data. mutable.
  T* MutableData(platform::Place place) {
    if (platform::is_gpu_place(place)) {
      return CUDAMutableData(place);
    } else {
      return data();
    }
  }

  // implicit cast operator. Vector can be cast to std::vector implicitly.
  operator std::vector<T>() const {
    std::vector<T> result;
    result.resize(size());
    std::copy(begin(), end(), result.begin());
    return result;
  }

  bool operator==(const Vector<T>& other) const {
    if (size() != other.size()) return false;
    auto it1 = cbegin();
    auto it2 = other.cbegin();
    for (; it1 < cend(); ++it1, ++it2) {
      if (*it1 != *it2) {
        return false;
      }
    }
    return true;
  }

 private:
  void InitEmpty() {
    size_ = 0;
    flag_ = kDataInCPU;
  }

  template <typename Iter>
  void InitByIter(size_t size, Iter begin, Iter end) {
    platform::Place cpu = platform::CPUPlace();
    T* ptr = this->cpu_vec_.template mutable_data<T>(
        framework::make_ddim({static_cast<int64_t>(size)}), cpu);
    for (size_t i = 0; i < size; ++i) {
      *ptr++ = *begin++;
    }
    flag_ = kDataInCPU | kDirty;
    size_ = size;
  }

  enum DataFlag {
    kDataInCPU = 0x01,
    kDataInCUDA = 0x02,
    // kDirty means the data has been changed in one device.
    kDirty = 0x10
  };

  void CopyToCPU() const {
    // COPY GPU Data To CPU
    TensorCopy(cuda_vec_, platform::CPUPlace(), &cpu_vec_);
    WaitPlace(cuda_vec_.place());
  }

  void MutableCPU() {
    if (IsInCUDA() && IsDirty()) {
      CopyToCPU();
    }
    flag_ = kDirty | kDataInCPU;
  }

  void ImmutableCUDA(platform::Place place) const {
    if (IsDirty()) {
      if (IsInCPU()) {
        TensorCopy(cpu_vec_, boost::get<platform::CUDAPlace>(place),
                   &cuda_vec_);
        WaitPlace(place);
        UnsetFlag(kDirty);
        SetFlag(kDataInCUDA);
      } else if (IsInCUDA() && !(place == cuda_vec_.place())) {
        framework::Tensor tmp;
        TensorCopy(cuda_vec_, boost::get<platform::CUDAPlace>(place), &tmp);
        WaitPlace(cuda_vec_.place());
        cuda_vec_.ShareDataWith(tmp);
        // Still dirty
      } else {
        // Dirty && DataInCUDA && Device is same
        // Do nothing
      }
    } else {
      if (!IsInCUDA()) {
        // Even data is not dirty. However, data is not in CUDA. Copy data.
        TensorCopy(cpu_vec_, boost::get<platform::CUDAPlace>(place),
                   &cuda_vec_);
        WaitPlace(place);
        SetFlag(kDataInCUDA);
      } else if (!(place == cuda_vec_.place())) {
        framework::Tensor tmp;
        WaitPlace(cuda_vec_.place());
        TensorCopy(cuda_vec_, boost::get<platform::CUDAPlace>(place), &tmp);
        WaitPlace(cuda_vec_.place());
        WaitPlace(place);
        cuda_vec_.ShareDataWith(tmp);
      } else {
        // Not Dirty && DataInCUDA && Device is same
        // Do nothing.
      }
    }
  }

  void ImmutableCPU() const {
    if (IsDirty() &&
        !IsInCPU()) {  // If data has been changed in CUDA, or CPU has no data.
      CopyToCPU();
      UnsetFlag(kDirty);
    }
    SetFlag(kDataInCPU);
  }

  void UnsetFlag(int flag) const { flag_ &= ~flag; }
  void SetFlag(int flag) const { flag_ |= flag; }

  bool IsDirty() const { return flag_ & kDirty; }

  bool IsInCUDA() const { return flag_ & kDataInCUDA; }

  bool IsInCPU() const { return flag_ & kDataInCPU; }

  static void WaitPlace(const platform::Place place) {
    if (platform::is_gpu_place(place)) {
      platform::DeviceContextPool::Instance()
          .Get(boost::get<platform::CUDAPlace>(place))
          ->Wait();
    }
  }

  static T& EmptyDummy() {
    static T dummy = T();
    return dummy;
  }

  mutable int flag_;
  mutable Tensor cpu_vec_;
  mutable Tensor cuda_vec_;
  size_t size_;
};

}  // namespace framework
}  // namespace paddle
