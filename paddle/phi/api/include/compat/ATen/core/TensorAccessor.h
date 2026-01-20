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
#include <algorithm>

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

// GenericPackedTensorAccessorBase stores sizes and strides internally
// (copies data instead of storing pointers, unlike TensorAccessor)
template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class GenericPackedTensorAccessorBase {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  GenericPackedTensorAccessorBase(PtrType data_ptr,
                                  const index_t* sizes_ptr,
                                  const index_t* strides_ptr)  // NOLINT
      : data_(data_ptr) {
    std::copy(sizes_ptr, sizes_ptr + N, std::begin(this->sizes_));
    std::copy(strides_ptr, strides_ptr + N, std::begin(this->strides_));
  }

  // Constructor for converting from int64_t to other index types
  template <typename source_index_t,
            class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  GenericPackedTensorAccessorBase(PtrType data_ptr,
                                  const source_index_t* sizes_ptr,
                                  const source_index_t* strides_ptr)  // NOLINT
      : data_(data_ptr) {
    for (size_t i = 0; i < N; ++i) {
      this->sizes_[i] = static_cast<index_t>(sizes_ptr[i]);
      this->strides_[i] = static_cast<index_t>(strides_ptr[i]);
    }
  }

  index_t stride(index_t i) const { return strides_[i]; }
  index_t size(index_t i) const { return sizes_[i]; }
  PtrType data() { return data_; }
  const PtrType data() const { return data_; }

 protected:
  PtrType data_;
  index_t sizes_[N];
  index_t strides_[N];
};

// GenericPackedTensorAccessor is used for packed tensor accessors
// It copies sizes and strides internally, unlike TensorAccessor
template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class GenericPackedTensorAccessor
    : public GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  GenericPackedTensorAccessor(PtrType data_ptr,
                              const index_t* sizes_ptr,
                              const index_t* strides_ptr)
      : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(
            data_ptr, sizes_ptr, strides_ptr) {}

  // Constructor for converting from int64_t to other index types
  template <typename source_index_t,
            class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  GenericPackedTensorAccessor(PtrType data_ptr,
                              const source_index_t* sizes_ptr,
                              const source_index_t* strides_ptr)
      : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(
            data_ptr, sizes_ptr, strides_ptr) {}
};

// Type aliases for PackedTensorAccessor32 and PackedTensorAccessor64
// Compatible with libtorch's naming convention
template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor32 =
    GenericPackedTensorAccessor<T, N, PtrTraits, int32_t>;

template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor64 =
    GenericPackedTensorAccessor<T, N, PtrTraits, int64_t>;

}  // namespace at
