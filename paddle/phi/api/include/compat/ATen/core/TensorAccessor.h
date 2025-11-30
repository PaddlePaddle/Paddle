// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

namespace at {
template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class TensorAccessorBase {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessorBase(PtrType data_,
                                     const index_t* sizes_,
                                     const index_t* strides_)  // NOLINT
      : data_(data_), sizes_(sizes_), strides_(strides_) {}    // NOLINT
  C10_HOST IntArrayRef sizes() const { return IntArrayRef(sizes_, N); }
  C10_HOST IntArrayRef strides() const { return IntArrayRef(strides_, N); }
  C10_HOST_DEVICE index_t stride(index_t i) const { return strides_[i]; }
  C10_HOST_DEVICE index_t size(index_t i) const { return sizes_[i]; }
  C10_HOST_DEVICE PtrType data() { return data_; }
  C10_HOST_DEVICE const PtrType data() const { return data_; }

 protected:
  PtrType data_;
  const index_t* sizes_;
  const index_t* strides_;
};

// The `TensorAccessor` is typically instantiated for CPU `Tensor`s using
// `Tensor.accessor<T, N>()`.
// For CUDA `Tensor`s, `GenericPackedTensorAccessor` is used on the host and
// only indexing on the device uses `TensorAccessor`s.
template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T, N, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(PtrType data_,
                                 const index_t* sizes_,
                                 const index_t* strides_)
      : TensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

  C10_HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](
      index_t i) {
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1);
  }

  C10_HOST_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](
      index_t i) const {
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1);
  }
};

template <typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T, 1, PtrTraits, index_t>
    : public TensorAccessorBase<T, 1, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(PtrType data_,
                                 const index_t* sizes_,
                                 const index_t* strides_)
      : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}
  C10_HOST_DEVICE T& operator[](index_t i) {
    return this->data_[this->strides_[0] * i];
  }
  C10_HOST_DEVICE const T& operator[](index_t i) const {
    return this->data_[this->strides_[0] * i];
  }
};

}  // namespace at
