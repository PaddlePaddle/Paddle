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
#include "paddle/framework/ddim.h"
#include "paddle/framework/enforce.h"
#include "paddle/framework/tensor_types.h"
#include "paddle/memory/memory.h"
#include "paddle/platform/place.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace framework {

class Tensor {
 public:
  Tensor() : offset_(0) {}

  template <typename T>
  const T* data() const {
    CheckDims<T>();
    return reinterpret_cast<const T*>(
        reinterpret_cast<uintptr_t>(holder_->ptr()) + offset_);
  }

  template <typename T>
  T* raw_data() const {
    CheckDims<T>();
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                offset_);
  }

  template <typename T>
  T* mutable_data(DDim dims, platform::Place place) {
    set_dims(dims);
    return mutable_data<T>(place);
  }

  template <typename T>
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
#ifdef __CUDACC__
        holder_.reset(new PlaceholderImpl<T, platform::GPUPlace>(
            boost::get<platform::GPUPlace>(place), product(dims_) * sizeof(T)));
#else
        PADDLE_ENFORCE(true, "'GPUPlace' is not supported in CPU only device.");
#endif
      } else {
        PADDLE_ENFORCE(true, "Unknown 'place'.");
      }
      offset_ = 0;
    }
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                offset_);
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor shaped(DDim new_dims) {
    Eigen::array<Eigen::DenseIndex, NDIMS> dims =
        paddle::framework::ToEigenDSizes<NDIMS>(new_dims);
    return typename TTypes<T, NDIMS>::Tensor(raw_data<T>(), dims);
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor tensor() {
    return typename TTypes<T, NDIMS>::Tensor(
        raw_data<T>(), paddle::framework::ToEigenDSizes<NDIMS>(dims_));
  }

  // flat to rank = 1
  template <typename T>
  typename TTypes<T>::Flat flat() {
    return shaped<T, 1>(make_ddim({static_cast<int>(product(dims_))}));
  }

  // to TensorType Vec
  template <typename T>
  typename TTypes<T>::Vec vec() {
    return tensor<T, 1>();
  }

  // to TensorType Matrix
  template <typename T>
  typename TTypes<T>::Matrix matrix() {
    return tensor<T, 2>();
  }

  // const versions of all the methods above.
  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor shaped(DDim new_dims) const {
    Eigen::array<Eigen::DenseIndex, NDIMS> dims =
        paddle::framework::ToEigenDSizes<NDIMS>(new_dims);
    return typename TTypes<T, NDIMS>::Tensor(data<T>(), dims);
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::ConstantTensor tensor() const {
    return typename TTypes<T, NDIMS>::Tensor(
        data<T>(), paddle::framework::ToEigenDSizes<NDIMS>(dims_));
  }

  template <typename T>
  typename TTypes<T>::ConstFlat flat() const {
    return shaped<T, 1>(make_ddim({static_cast<int>(product(dims_))}));
  }

  template <typename T>
  typename TTypes<T>::ConstVec vec() const {
    return tensor<T, 1>();
  }

  template <typename T>
  typename TTypes<T>::ConstMatrix matrix() const {
    return tensor<T, 2>();
  }

  template <typename T>
  void ShareDataFrom(const Tensor& src) {
    src.CheckDims<T>();
    holder_ = src.holder_;
    set_dims(src.dims());
    offset_ = src.offset_;
  }

  template <typename T>
  void CopyFrom(const Tensor& src, platform::Place dst_place) {
    PADDLE_ENFORCE(platform::is_cpu_place(src.holder_->place()) &&
                       platform::is_cpu_place(dst_place),
                   "Tensor::CopyFrom only support CPU now.");
    src.CheckDims<T>();
    size_t size = product(src.dims_) * sizeof(T);
    set_dims(src.dims());
    const void* src_ptr = static_cast<const void*>(src.data<T>());
    void* dst_ptr = static_cast<void*>(mutable_data<T>(dst_place));
    memcpy(dst_ptr, src_ptr, size);
  }

  template <typename T>
  Tensor Slice(const int& begin_idx, const int& end_idx) const {
    CheckDims<T>();
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
    DDim dst_dims = dims_;
    dst_dims[0] = end_idx - begin_idx;
    dst.set_dims(dst_dims);
    dst.offset_ = offset_ + begin_idx * base * sizeof(T);
    return dst;
  }

  void set_dims(const DDim& dims) {
    if (dims == dims_) {
      return;
    }
    dims_ = dims;
  }

  DDim dims() const { return dims_; }

 private:
  // Placeholder hides type T, so it doesn't appear as a template
  // parameter of Variable.
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual void* ptr() const = 0;
    virtual platform::Place place() const = 0;
    virtual size_t size() const = 0;
  };

  template <typename T, typename PlaceType>
  struct PlaceholderImpl : public Placeholder {
   private:
    template <typename PType>
    class Deleter {
     public:
      Deleter(PType place) : place_(place) {}
      void operator()(T* ptr) { memory::Free(place_, static_cast<void*>(ptr)); }

     private:
      PType place_;
    };

   public:
    PlaceholderImpl(PlaceType place, size_t size)
        : ptr_(static_cast<T*>(memory::Alloc(place, size)),
               Deleter<PlaceType>(place)),
          place_(place),
          size_(size) {}

    virtual void* ptr() const { return static_cast<void*>(ptr_.get()); }
    virtual size_t size() const { return size_; }
    virtual platform::Place place() const { return place_; }

    std::unique_ptr<T, Deleter<PlaceType>> ptr_;
    platform::Place place_;  // record the place of ptr_.
    size_t size_;            // size of the memory block.
  };

  template <typename T>
  inline void CheckDims() const {
    PADDLE_ENFORCE(holder_ != nullptr,
                   "Tenosr holds no memory. Call Tensor::mutable_data first.");
    PADDLE_ENFORCE(holder_->size() >= product(dims_) * sizeof(T) + offset_,
                   "Tensor's dims_ is out of bound. Call Tensor::mutable_data "
                   "first to re-allocate memory.");
  }

  std::shared_ptr<Placeholder> holder_;  // holds the memory block if allocated.
  DDim dims_;
  size_t offset_;  // marks the begin of tensor data area.
};

}  // namespace framework
}  // namespace paddle
