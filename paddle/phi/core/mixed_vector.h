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
#include <mutex>  // NOLINT
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/utils/none.h"
#include "paddle/utils/optional.h"

namespace phi {

template <class T>
using Vector = std::vector<T>;

inline paddle::optional<phi::GPUPlace> OptionalCUDAPlace(
    const phi::Allocator::AllocationPtr &gpu_) {
  return gpu_ == nullptr ? paddle::none
                         : paddle::optional<phi::GPUPlace>(gpu_->place());
}

// Vector<T> implements the std::vector interface, and can get Data or
// MutableData from any place. The data will be synced implicitly inside.
template <typename T>
class MixVector {
 public:
  using value_type = T;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

 private:
  // The actual class to implement vector logic
  class VectorData {
   public:
    template <typename U>
    explicit VectorData(std::vector<U> *dat) : cpu_(dat), flag_(kDataInCPU) {}
    ~VectorData() {}

    VectorData(const VectorData &o) = delete;

    VectorData &operator=(const VectorData &o) = delete;

    T &operator[](size_t i) {
      MutableCPU();
      return (*cpu_)[i];
    }

    const T &operator[](size_t i) const {
      ImmutableCPU();
      return (*cpu_)[i];
    }

    size_t size() const { return (*cpu_).size(); }

    iterator begin() {
      MutableCPU();
      return (*cpu_).begin();
    }

    iterator end() {
      MutableCPU();
      return (*cpu_).end();
    }

    T &front() {
      MutableCPU();
      return (*cpu_).front();
    }

    T &back() {
      MutableCPU();
      return (*cpu_).back();
    }

    const_iterator begin() const {
      ImmutableCPU();
      return (*cpu_).begin();
    }

    const_iterator end() const {
      ImmutableCPU();
      return (*cpu_).end();
    }

    const T &back() const {
      ImmutableCPU();
      return (*cpu_).back();
    }

    T *data() { return cpu_->data(); }

    const T *data() const { return cpu_->data(); }

    const T &front() const {
      ImmutableCPU();
      return (*cpu_).front();
    }

    // assign this from iterator.
    // NOTE: the iterator must support `end-begin`
    template <typename Iter>
    void assign(Iter begin, Iter end) {
      MutableCPU();
      (*cpu_).assign(begin, end);
    }

    // push_back. If the previous capacity is not enough, the memory will
    // double.
    void push_back(T elem) {
      MutableCPU();
      (*cpu_).push_back(elem);
    }

    // extend a vector by iterator.
    // NOTE: the iterator must support end-begin
    template <typename It>
    void Extend(It begin, It end) {
      MutableCPU();
      auto out_it = std::back_inserter<std::vector<T>>(*(this->cpu_));
      std::copy(begin, end, out_it);
    }

    // resize the vector
    void resize(size_t size) {
      MutableCPU();
      (*cpu_).resize(size);
    }

    // get cuda ptr. immutable
    const T *CUDAData(phi::Place place) const {
      PADDLE_ENFORCE_EQ(
          place.GetType() == phi::AllocationType::GPU,
          true,
          phi::errors::Unavailable(
              "Place mismatch, CUDA Data must be on CUDA place."));
      ImmutableCUDA(place);
      return reinterpret_cast<T *>(gpu_->ptr());
    }

    // get cuda ptr. mutable
    T *CUDAMutableData(phi::Place place) {
      const T *ptr = CUDAData(place);
      flag_ = kDirty | kDataInCUDA;
      return const_cast<T *>(ptr);
    }

    // clear
    void clear() {
      (*cpu_).clear();
      flag_ = kDirty | kDataInCPU;
    }

    std::vector<T> *get_vector() { return cpu_; }

    size_t capacity() const { return (*cpu_).capacity(); }

    // reserve data
    void reserve(size_t size) const { (*cpu_).reserve(size); }

    std::mutex &Mutex() const { return mtx_; }

    paddle::optional<phi::GPUPlace> CUDAPlace() const {
      return OptionalCUDAPlace(gpu_);
    }

    void MutableCPU() {
      if (IsInCUDA() && IsDirty()) {
        CopyToCPU();
      }
      flag_ = kDirty | kDataInCPU;
    }

   private:
    enum DataFlag {
      kDataInCPU = 0x01,
      kDataInCUDA = 0x02,
      // kDirty means the data has been changed in one device.
      kDirty = 0x10
    };

    void CopyToCPU() const;

    void ImmutableCUDA(phi::Place place) const {
      if (IsDirty()) {
        if (IsInCPU()) {
          CopyCPUDataToCUDA(place);
          UnsetFlag(kDirty);
          SetFlag(kDataInCUDA);
        } else if (IsInCUDA() && !(place == gpu_->place())) {
          PADDLE_THROW(
              phi::errors::Unavailable("Unexpected data place mismatch."));
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
        } else if (!(place == gpu_->place())) {
          PADDLE_THROW(
              phi::errors::Unavailable("Unexpected data place mismatch."));
        } else {
          // Not Dirty && DataInCUDA && Device is same
          // Do nothing.
        }
      }
    }

    void CopyCPUDataToCUDA(const phi::Place &place) const;

    void ImmutableCPU() const {
      if (IsDirty() && !IsInCPU()) {  // If data has been changed in CUDA, or
                                      // CPU has no data.
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

    std::vector<T> *cpu_;
    mutable phi::Allocator::AllocationPtr gpu_;
    mutable size_t gpu_memory_size_{0};
    mutable int flag_;

    mutable std::mutex mtx_;
  };

 public:
  // implicit cast from std::vector.
  template <typename U>
  MixVector(const std::vector<U> *dat) {  // NOLINT
    m_.reset(new VectorData(const_cast<std::vector<U> *>(dat)));
  }

  // Copy ctor
  MixVector(const MixVector<T> &other) = delete;

  // Copy operator
  MixVector<T> &operator=(const MixVector<T> &other) = delete;

  // Move ctor
  MixVector(MixVector<T> &&other) = delete;

  // CPU data access method. Mutable.
  T &operator[](size_t i) { return (*m_)[i]; }

  // CPU data access method. Immutable.
  const T &operator[](size_t i) const { return (*m_)[i]; }

  // std::vector iterator methods. Based on CPU data access method
  size_t size() const { return m_->size(); }

  iterator begin() { return m_->begin(); }

  iterator end() { return m_->end(); }

  T &front() { return m_->front(); }

  T &back() { return m_->back(); }

  const_iterator begin() const { return m_->begin(); }

  const_iterator end() const { return m_->end(); }

  const_iterator cbegin() const { return begin(); }

  const_iterator cend() const { return end(); }

  const T &back() const { return m_->back(); }

  T *data() { return m_->data(); }

  const T *data() const { return m_->data(); }

  const T &front() const { return m_->front(); }
  // end of std::vector iterator methods

  // assign this from iterator.
  // NOTE: the iterator must support `end-begin`
  template <typename Iter>
  void assign(Iter begin, Iter end) {
    m_->assign(begin, end);
  }

  // push_back. If the previous capacity is not enough, the memory will
  // double.
  void push_back(T elem) { m_->push_back(elem); }

  // extend a vector by iterator.
  // NOTE: the iterator must support end-begin
  template <typename It>
  void Extend(It begin, It end) {
    m_->Extend(begin, end);
  }

  // resize the vector
  void resize(size_t size) {
    if (m_->size() != size) {
      m_->resize(size);
    }
  }

  // get cuda ptr. immutable
  const T *CUDAData(phi::Place place) const {
    {
      phi::GPUPlace p(place.GetDeviceId());
      auto &mtx = m_->Mutex();
      std::lock_guard<std::mutex> guard(mtx);
      auto cuda_place = m_->CUDAPlace();
      if (cuda_place == paddle::none || cuda_place == p) {
        return m_->CUDAData(place);
      }
    }
    m_->MutableCPU();
    m_.reset(new VectorData(m_->get_vector()));
    return CUDAData(place);
  }

  // get cuda ptr. mutable
  T *CUDAMutableData(phi::Place place) {
    {
      phi::GPUPlace p(place.GetDeviceId());
      auto &mtx = m_->Mutex();
      std::lock_guard<std::mutex> guard(mtx);
      auto cuda_place = m_->CUDAPlace();
      if (cuda_place == paddle::none || cuda_place == p) {
        return m_->CUDAMutableData(place);
      }
    }
    m_->MutableCPU();
    m_.reset(new VectorData(m_->get_vector()));
    return CUDAMutableData(place);
  }

  // clear
  void clear() { m_->clear(); }

  size_t capacity() const { return m_->capacity(); }

  // reserve data
  void reserve(size_t size) { m_->reserve(size); }

  // the unify method to access CPU or CUDA data. immutable.
  const T *Data(phi::Place place) const {
    if (place.GetType() == phi::AllocationType::GPU) {
      return CUDAData(place);
    } else {
      return data();
    }
  }

  // the unify method to access CPU or CUDA data. mutable.
  T *MutableData(phi::Place place) {
    if (place.GetType() == phi::AllocationType::GPU) {
      return CUDAMutableData(place);
    } else {
      return data();
    }
  }

  void CopyToCPU() { m_->MutableCPU(); }

  const void *Handle() const { return m_.get(); }

 private:
  mutable std::unique_ptr<VectorData> m_;
};

};  // namespace phi
