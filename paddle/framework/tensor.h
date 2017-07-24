/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <cstdint>
#include <cstring>
#include <memory>
#include <typeindex>
#include "paddle/framework/ddim.h"
#include "paddle/memory/memory.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/place.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace pybind {
namespace details {  // forward declare
template <bool less, size_t i, typename... args>
struct CastToPyBufferImpl;
}  // namespace details
}  // namespace pybind
namespace framework {

class Tensor {
  template <bool less, size_t i, typename... args>
  friend struct paddle::pybind::details::CastToPyBufferImpl;

  template <typename T, size_t D, int MajorType, typename IndexType>
  friend struct EigenTensor;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenVector;

 public:
  Tensor() : offset_(0) {}

  template <typename T>
  const T* data() const {
    EnforceSufficientMemory<T>();
    return reinterpret_cast<const T*>(
        reinterpret_cast<uintptr_t>(holder_->ptr()) + offset_);
  }

  template <typename T>
  T* data() {
    EnforceSufficientMemory<T>();
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                offset_);
  }

  template <typename T,  // must be POD types
            typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
  T* mutable_data(DDim dims, platform::Place place) {
    Resize(dims);
    return mutable_data<T>(place);
  }

  template <typename T,  // must be POD types
            typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
  T* mutable_data(platform::Place place) {
    PADDLE_ENFORCE(product(dims_) > 0,
                   "Tensor's numel must be larger than zero to call "
                   "Tensor::mutable_data. Call Tensor::set_dim first.");
    if (holder_ == nullptr ||
        !(holder_->place() ==
          place) /* some versions of boost::variant don't have operator!= */
        || holder_->size() < product(dims_) * sizeof(T) + offset_) {
      if (platform::is_cpu_place(place)) {
        holder_.reset(new PlaceholderImpl<T, platform::CPUPlace>(
            boost::get<platform::CPUPlace>(place), product(dims_) * sizeof(T)));
      } else if (platform::is_gpu_place(place)) {
#ifdef PADDLE_ONLY_CPU
        PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
#else
        holder_.reset(new PlaceholderImpl<T, platform::GPUPlace>(
            boost::get<platform::GPUPlace>(place), product(dims_) * sizeof(T)));
#endif
      } else {
        PADDLE_THROW("Unknown 'place'.");
      }
      offset_ = 0;
    }
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                offset_);
  }

  template <typename T>
  void ShareDataWith(const Tensor& src) {
    src.EnforceSufficientMemory<T>();
    *this = src;
  }

  template <typename T>
  void CopyFrom(const Tensor& src, platform::Place dst_place) {
    PADDLE_ENFORCE(platform::is_cpu_place(src.holder_->place()) &&
                       platform::is_cpu_place(dst_place),
                   "Tensor::CopyFrom only support CPU now.");
    src.EnforceSufficientMemory<T>();
    size_t size = product(src.dims_) * sizeof(T);
    Resize(src.dims());
    const void* src_ptr = static_cast<const void*>(src.data<T>());
    void* dst_ptr = static_cast<void*>(mutable_data<T>(dst_place));
    memcpy(dst_ptr, src_ptr, size);
  }

  template <typename T>
  Tensor Slice(const int& begin_idx, const int& end_idx) const {
    EnforceSufficientMemory<T>();
    PADDLE_ENFORCE(begin_idx >= 0, "Slice begin index is less than zero.");
    PADDLE_ENFORCE(end_idx <= dims_[0], "Slice end index is out of bound.");
    PADDLE_ENFORCE(begin_idx < end_idx,
                   "Begin index must be less than end index.");
    PADDLE_ENFORCE(dims_[0] != 1, "Can not slice a tensor with dims_[0] = 1.");
    int base = product(dims_) / dims_[0];
    Tensor dst;
    dst.holder_ = holder_;
    DDim dst_dims = dims_;
    dst_dims[0] = end_idx - begin_idx;
    dst.Resize(dst_dims);
    dst.offset_ = offset_ + begin_idx * base * sizeof(T);
    return dst;
  }

  void Resize(const DDim& dims) { dims_ = dims; }

  const DDim& dims() const { return dims_; }

 private:
  // Placeholder hides type T, so it doesn't appear as a template
  // parameter of Variable.
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual void* ptr() const = 0;
    virtual platform::Place place() const = 0;
    virtual size_t size() const = 0;
    virtual std::type_index type() const = 0;
  };

  template <typename T, typename PlaceType>
  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(PlaceType place, size_t size)
        : ptr_(static_cast<T*>(memory::Alloc(place, size)),
               memory::PODDeleter<T, PlaceType>(place)),
          place_(place),
          size_(size) {}

    virtual void* ptr() const { return static_cast<void*>(ptr_.get()); }
    virtual size_t size() const { return size_; }
    virtual paddle::platform::Place place() const { return place_; }
    virtual std::type_index type() const { return std::type_index(typeid(T)); }

    std::unique_ptr<T, memory::PODDeleter<T, PlaceType>> ptr_;
    platform::Place place_;  // record the place of ptr_.
    size_t size_;            // size of the memory block.
  };

  template <typename T>
  inline void EnforceSufficientMemory() const {
    PADDLE_ENFORCE(holder_ != nullptr,
                   "Tenosr holds no memory. Call Tensor::mutable_data first.");
    PADDLE_ENFORCE(holder_->size() >= product(dims_) * sizeof(T) + offset_,
                   "Tensor's dims_ is out of bound. Call Tensor::mutable_data "
                   "first to re-allocate memory.");
  }

  std::shared_ptr<Placeholder> holder_;  // holds the memory block if allocated.
  DDim dims_;
  // A PlaceHolder may be shared by more than one tensor. Some of them may be
  // slices of the others. So the offset_ is introduced here to indicate the
  // byte offset between PlaceHolder::ptr_ and where tensor's data really
  // begins.
  size_t offset_;
};

}  // namespace framework
}  // namespace paddle
