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

#include <memory>
#include <type_traits>
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
  template <typename T>
  T* data() const {
    PADDLE_ENFORCE(holder_ != nullptr,
                   "Tensor::data must be called after Tensor::mutable_data.");
    return static_cast<T*>(holder_->Ptr());
  }

  template <typename T,  // must be POD types
            typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
  T* mutable_data(DDim dims, paddle::platform::Place place) {
    if (holder_ == nullptr ||
        !(holder_->Place() ==
          place) /* some versions of boost::variant don't have operator!= */
        || holder_->Size() < product(dims) * sizeof(T)) {
      holder_.reset(new PlaceholderImpl<T>(place, product(dims) * sizeof(T)));
      dims_ = dims;
    }
    return static_cast<T*>(holder_->Ptr());
  }

  DDim dim() const { return dims_; }

  size_t NumElements() const { return product(dims_); }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor shaped(DDim new_dims) {
    Eigen::array<Eigen::DenseIndex, NDIMS> dims =
        paddle::framework::ToEigenDSizes<NDIMS>(new_dims);
    return typename TTypes<T, NDIMS>::Tensor(data<T>(), dims);
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor tensor() {
    return typename TTypes<T, NDIMS>::Tensor(
        data<T>(), paddle::framework::ToEigenDSizes<NDIMS>(dims_));
  }

  // flat to rank = 1
  template <typename T>
  typename TTypes<T>::Flat flat() {
    return shaped<T, 1>({NumElements()});
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
  typename TTypes<T, NDIMS>::ConstantTensor tensor() const {
    return typename TTypes<T, NDIMS>::Tensor(
        data<T>(), paddle::framework::ToEigenDSizes<NDIMS>(dims_));
  }

  template <typename T>
  typename TTypes<T>::ConstFlat flat() const {
    return shaped<T, 1>({NumElements()});
  }

  template <typename T>
  typename TTypes<T>::ConstVec vec() const {
    return tensor<T, 1>();
  }

  template <typename T>
  typename TTypes<T>::ConstMatrix matrix() const {
    return tensor<T, 2>();
  }

 private:
  // Placeholder hides type T, so it doesn't appear as a template
  // parameter of Variable.
  struct Placeholder {
    virtual ~Placeholder() {}
    virtual void* Ptr() const = 0;
    virtual paddle::platform::Place Place() const = 0;
    virtual size_t Size() const = 0;
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

    std::unique_ptr<T, Deleter> ptr_;
    paddle::platform::Place place_;  // record the place of ptr_.
    size_t size_;                    // size of the memory block.
  };

  std::shared_ptr<Placeholder> holder_;  // holds the memory block if allocated.
  DDim dims_;
};

}  // namespace framework
}  // namespace paddle
