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
#include <type_traits>
#include "paddle/framework/ddim.h"
#include "paddle/framework/enforce.h"
#include "paddle/memory/memory.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace framework {

class Tensor {
 public:
  Tensor() : offset_(0) {}

  explicit Tensor(const DDim& dims) : dims_(dims), offset_(0) {}

  template <typename T>
  const T* data() const {
    PADDLE_ENFORCE(
        holder_ != nullptr,
        "Tenosr has not been initialized. Call Tensor::mutable_data first.");
    return reinterpret_cast<const T*>(
        reinterpret_cast<uintptr_t>(holder_->Ptr()) + offset_);
  }

  template <typename T,  // must be POD types
            typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
  T* mutable_data(DDim dims, paddle::platform::Place place) {
    dims_ = dims;
    return mutable_data<T>(place);
  }

  template <typename T,  // must be POD types
            typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
  T* mutable_data(paddle::platform::Place place) {
    if (holder_ == nullptr ||
        !(holder_->Place() ==
          place) /* some versions of boost::variant don't have operator!= */
        || holder_->Size() < product(dims_) * sizeof(T) + offset_) {
      holder_.reset(new PlaceholderImpl<T>(place, product(dims_) * sizeof(T)));
      offset_ = 0;
    }
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(holder_->Ptr()) +
                                offset_);
  }

  void ShareDataFrom(const Tensor& src) {
    PADDLE_ENFORCE(src.holder_ != nullptr,
                   "Can not share data from an uninitialized tensor.");
    holder_ = src.holder_;
    dims_ = src.dims_;
    offset_ = src.offset_;
  }

  void CopyFrom(const Tensor& src, paddle::platform::Place dst_place) {
    PADDLE_ENFORCE(src.holder_ != nullptr,
                   "Can not copy from an uninitialized tensor.");
    size_t size = product(src.dims()) * src.holder_->TypeSize();
    holder_.reset(src.holder_->Clone(src.offset_, size, dst_place));
    dims_ = src.dims();
    offset_ = 0;
  }

  Tensor Slice(const int& begin_idx, const int& end_idx) const {
    PADDLE_ENFORCE(holder_ != nullptr,
                   "The sliced tenosr has not been initialized.");
    PADDLE_ENFORCE(begin_idx >= 0 && end_idx <= dims_[0],
                   "Slice index is less than zero or out of bound.");
    PADDLE_ENFORCE(begin_idx < end_idx,
                   "Begin index must be less than end index.");
    PADDLE_ENFORCE(dims_[0] != 1, "Can not slice a tensor with dims_[0] = 1.");
    std::vector<int> d = vectorize(dims_);
    int base = 1;
    for (size_t i = 1; i < d.size(); ++i) {
      base *= d[i];
    }
    Tensor dst;
    dst.holder_ = holder_;
    dst.dims_ = dims_;
    dst.dims_[0] = end_idx - begin_idx;
    dst.offset_ = offset_ + begin_idx * base * holder_->TypeSize();
    return dst;
  }

  DDim dims() const { return dims_; }

 private:
  // Placeholder hides type T, so it doesn't appear as a template
  // parameter of Variable.
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual void* Ptr() const = 0;
    virtual paddle::platform::Place Place() const = 0;
    virtual size_t Size() const = 0;
    virtual size_t TypeSize() const = 0;
    virtual Placeholder* Clone(size_t begin, size_t size,
                               paddle::platform::Place place) const = 0;
  };

  template <typename T>
  struct PlaceholderImpl : public Placeholder {
   private:
    class Deleter {
     public:
      Deleter(platform::Place place) : place_(place) {}
      void operator()(T* ptr) {
        paddle::memory::Free(place_, static_cast<void*>(ptr));
      }

     private:
      paddle::platform::Place place_;
    };

   public:
    PlaceholderImpl(paddle::platform::Place place, size_t size)
        : ptr_(static_cast<T*>(paddle::memory::Alloc(place, size)),
               Deleter(place)),
          place_(place),
          size_(size) {}

    virtual void* Ptr() const { return static_cast<void*>(ptr_.get()); }
    virtual size_t Size() const { return size_; }
    virtual paddle::platform::Place Place() const { return place_; }
    virtual size_t TypeSize() const { return sizeof(T); }
    // TODO: Clone only support CPU now. GPU support is needed.
    virtual Placeholder* Clone(size_t begin, size_t size,
                               paddle::platform::Place place) const {
      PADDLE_ENFORCE(paddle::platform::is_cpu_place(place_) &&
                         paddle::platform::is_cpu_place(place),
                     "PlaceholderImpl::Clone only support CPU now.");
      PlaceholderImpl<T>* dst = new PlaceholderImpl<T>(place, size);
      void* begin_ptr =
          reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(Ptr()) + begin);
      memcpy(dst->Ptr(), begin_ptr, size);
      return dst;
    }

    std::unique_ptr<T, Deleter> ptr_;
    paddle::platform::Place place_;  // record the place of ptr_.
    size_t size_;                    // size of the memory block.
  };

  std::shared_ptr<Placeholder> holder_;  // holds the memory block if allocated.
  DDim dims_;
  size_t offset_;  // marks the begin of tensor data area.
};

}  // namespace framework
}  // namespace paddle
