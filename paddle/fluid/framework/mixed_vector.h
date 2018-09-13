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
#include <memory>
#include <vector>
#include "paddle/fluid/memory/memcpy.h"

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"

#include "glog/logging.h"

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_CUDA)
// Vector<T> implements the std::vector interface, and can get Data or
// MutableData from any place. The data will be synced implicitly inside.
template <typename T>
class Vector {
 public:
  using value_type = T;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  // Default ctor. Create empty Vector
  Vector() { InitEmpty(); }

  // Fill vector with value. The vector size is `count`.
  explicit Vector(size_t count, const T &value = T()) : cpu_vec_(count, value) {
    InitEmpty();
  }

  // Ctor with init_list
  Vector(std::initializer_list<T> init) : cpu_vec_(init) { InitEmpty(); }

  // implicit cast from std::vector.
  template <typename U>
  Vector(const std::vector<U> &dat) : cpu_vec_(dat) {  // NOLINT
    InitEmpty();
  }

  // Copy ctor
  Vector(const Vector<T> &other) { this->operator=(other); }

  // Copy operator
  Vector<T> &operator=(const Vector<T> &other) {
    if (other.size() != 0) {
      this->InitByIter(other.size(), other.begin(), other.end());
    } else {
      InitEmpty();
    }
    return *this;
  }

  // Move ctor
  Vector(Vector<T> &&other) {
    this->flag_ = other.flag_;
    if (other.cuda_vec_.memory_size()) {
      this->cuda_vec_.ShareDataWith(other.cuda_vec_);
    }
    if (!other.cpu_vec_.empty()) {
      this->cpu_vec_ = std::move(other.cpu_vec_);
    }
  }

  // CPU data access method. Mutable.
  T &operator[](size_t i) {
    MutableCPU();
    return cpu_vec_[i];
  }

  // CPU data access method. Immutable.
  const T &operator[](size_t i) const {
    ImmutableCPU();
    return cpu_vec_[i];
  }

  // std::vector iterator methods. Based on CPU data access method
  size_t size() const { return cpu_vec_.size(); }

  iterator begin() {
    MutableCPU();
    return cpu_vec_.begin();
  }

  iterator end() {
    MutableCPU();
    return cpu_vec_.end();
  }

  T &front() {
    MutableCPU();
    return cpu_vec_.front();
  }

  T &back() {
    MutableCPU();
    return cpu_vec_.back();
  }

  const_iterator begin() const {
    ImmutableCPU();
    return cpu_vec_.begin();
  }

  const_iterator end() const {
    ImmutableCPU();
    return cpu_vec_.end();
  }

  const_iterator cbegin() const { return begin(); }

  const_iterator cend() const { return end(); }

  const T &back() const {
    ImmutableCPU();
    return cpu_vec_.back();
  }

  T *data() {
    MutableCPU();
    return &(*this)[0];
  }

  const T *data() const {
    ImmutableCPU();
    return &(*this)[0];
  }

  const T &front() const {
    ImmutableCPU();
    return cpu_vec_.front();
  }
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
    MutableCPU();
    cpu_vec_.push_back(elem);
  }

  // extend a vector by iterator.
  // NOTE: the iterator must support end-begin
  template <typename It>
  void Extend(It begin, It end) {
    MutableCPU();
    cpu_vec_.reserve((end - begin) + cpu_vec_.size());
    std::copy(begin, end, cpu_vec_.begin());
  }

  // resize the vector
  void resize(size_t size) {
    MutableCPU();
    cpu_vec_.resize(size);
  }

  // get cuda ptr. immutable
  const T *CUDAData(platform::Place place) const {
    PADDLE_ENFORCE(platform::is_gpu_place(place),
                   "CUDA Data must on CUDA place");
    ImmutableCUDA(place);
    return cuda_vec_.data<T>();
  }

  // get cuda ptr. mutable
  T *CUDAMutableData(platform::Place place) {
    const T *ptr = CUDAData(place);
    flag_ = kDirty | kDataInCUDA;
    return const_cast<T *>(ptr);
  }

  // clear
  void clear() {
    cpu_vec_.clear();
    flag_ = kDirty | kDataInCPU;
  }

  size_t capacity() const { return cpu_vec_.capacity(); }

  // reserve data
  void reserve(size_t size) { cpu_vec_.reserve(size); }

  // the unify method to access CPU or CUDA data. immutable.
  const T *Data(platform::Place place) const {
    if (platform::is_gpu_place(place)) {
      return CUDAData(place);
    } else {
      return data();
    }
  }

  // the unify method to access CPU or CUDA data. mutable.
  T *MutableData(platform::Place place) {
    if (platform::is_gpu_place(place)) {
      return CUDAMutableData(place);
    } else {
      return data();
    }
  }

  // implicit cast operator. Vector can be cast to std::vector implicitly.
  operator std::vector<T>() const {
    ImmutableCPU();
    return cpu_vec_;
  }

  bool operator==(const Vector<T> &other) const {
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
  void InitEmpty() { flag_ = kDataInCPU; }

  template <typename Iter>
  void InitByIter(size_t size, Iter begin, Iter end) {
    cpu_vec_.resize(size);
    std::copy(begin, end, cpu_vec_.begin());
    flag_ = kDataInCPU | kDirty;
  }

  enum DataFlag {
    kDataInCPU = 0x01,
    kDataInCUDA = 0x02,
    // kDirty means the data has been changed in one device.
    kDirty = 0x10
  };

  void CopyToCPU() const {
    // COPY GPU Data To CPU
    void *src = cuda_vec_.data<void>();
    void *dst = cpu_vec_.data();
    memory::Copy(platform::CPUPlace(), dst,
                 boost::get<platform::CUDAPlace>(cuda_vec_.place()), src,
                 cuda_vec_.memory_size(), nullptr);
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
        CopyCPUDataToCUDA(place);
        UnsetFlag(kDirty);
        SetFlag(kDataInCUDA);
      } else if (IsInCUDA() && !(place == cuda_vec_.place())) {
        CopyCUDADataToAnotherPlace(place);
        // Still dirty
      } else {
        // Dirty && DataInCUDA && Device is same
        // Do nothing
      }
    } else {
      if (!IsInCUDA()) {
        // Even data is not dirty. However, data is not in CUDA. Copy data.
        CopyCPUDataToCUDA(place);
        SetFlag(kDataInCUDA);
      } else if (!(place == cuda_vec_.place())) {
        CopyCUDADataToAnotherPlace(place);
      } else {
        // Not Dirty && DataInCUDA && Device is same
        // Do nothing.
      }
    }
  }
  void CopyCUDADataToAnotherPlace(const platform::Place &place) const {
    Tensor tmp;
    tmp.Resize(cuda_vec_.dims());

    const void *src = cuda_vec_.data<T>();
    void *dst = tmp.mutable_data<T>(place);

    memory::Copy(boost::get<platform::CUDAPlace>(place), dst,
                 boost::get<platform::CUDAPlace>(cuda_vec_.place()), src,
                 cuda_vec_.memory_size(), nullptr);
    cuda_vec_.ShareDataWith(tmp);
  }
  void CopyCPUDataToCUDA(const platform::Place &place) const {
    void *src = cpu_vec_.data();
    cuda_vec_.Resize(
        framework::make_ddim({static_cast<int64_t>(cpu_vec_.size())}));
    void *dst = cuda_vec_.mutable_data<T>(place);
    memory::Copy(boost::get<platform::CUDAPlace>(place), dst,
                 platform::CPUPlace(), src, cuda_vec_.memory_size(), nullptr);
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

  mutable int flag_;
  mutable std::vector<T> cpu_vec_;
  mutable Tensor cuda_vec_;
};

#else  // PADDLE_WITH_CUDA

template <typename T>
class CPUVector : public std::vector<T, std::allocator<T>> {
 public:
  CPUVector() : std::vector<T>() {}
  CPUVector(size_t count, const T &value = T())  // NOLINT
      : std::vector<T>(count, value) {}
  CPUVector(std::initializer_list<T> init) : std::vector<T>(init) {}
  CPUVector(const std::vector<T> &other) : std::vector<T>(other) {}  // NOLINT
  CPUVector(const CPUVector<T> &other) : std::vector<T>(other) {}
  CPUVector(CPUVector<T> &&other) : std::vector<T>(std::move(other)) {}
  CPUVector(std::vector<T> &&other)  // NOLINT
      : std::vector<T>(std::move(other)) {}
  CPUVector &operator=(const CPUVector &other) {
    this->assign(other.begin(), other.end());
    return *this;
  }
  CPUVector &operator=(const std::vector<T> &other) {
    this->assign(other.begin(), other.end());
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os, const CPUVector<T> &other) {
    std::stringstream ss;
    for (auto v : other) {
      os << v << " ";
    }
    return os;
  }

  T &operator[](size_t id) { return this->at(id); }

  const T &operator[](size_t id) const { return this->at(id); }

  template <typename D>
  void Extend(const D &begin, const D &end) {
    this->reserve(this->size() + size_t(end - begin));
    this->insert(this->end(), begin, end);
  }
};

template <typename T>
using Vector = CPUVector<T>;

#endif  // PADDLE_WITH_CUDA

};  // namespace framework
}  // namespace paddle
