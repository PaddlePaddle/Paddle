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

// GenericPackedTensorAccessorBase and GenericPackedTensorAccessor are used on
// for CUDA `Tensor`s on the host and as In contrast to `TensorAccessor`s, they
// copy the strides and sizes on instantiation (on the host) in order to
// transfer them on the device when calling kernels. On the device, indexing of
// multidimensional tensors gives to `TensorAccessor`s. Use RestrictPtrTraits as
// PtrTraits if you want the tensor's data pointer to be marked as __restrict__.
// Instantiation from data, sizes, strides is only needed on the host and
// std::copy isn't available on the device, so those functions are host only.
template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class GenericPackedTensorAccessorBase {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  C10_HOST GenericPackedTensorAccessorBase(PtrType data_arg,
                                           const index_t* sizes_arg,
                                           const index_t* strides_arg)
      : data_(data_arg) {
    std::copy(sizes_arg, sizes_arg + N, std::begin(this->sizes_));
    std::copy(strides_arg, strides_arg + N, std::begin(this->strides_));
  }

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <typename source_index_t,
            class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  C10_HOST GenericPackedTensorAccessorBase(PtrType data_arg,
                                           const source_index_t* sizes_arg,
                                           const source_index_t* strides_arg)
      : data_(data_arg) {
    for (size_t i = 0; i < N; ++i) {
      this->sizes_[i] = sizes_arg[i];
      this->strides_[i] = strides_arg[i];
    }
  }

  C10_HOST_DEVICE index_t stride(index_t i) const { return strides_[i]; }
  C10_HOST_DEVICE index_t size(index_t i) const { return sizes_[i]; }
  C10_HOST_DEVICE PtrType data() { return data_; }
  C10_HOST_DEVICE const PtrType data() const { return data_; }

 protected:
  PtrType data_;
  // NOLINTNEXTLINE
  index_t sizes_[N];
  // NOLINTNEXTLINE
  index_t strides_[N];
  C10_HOST void bounds_check_(index_t i) const {
    TORCH_CHECK_INDEX(0 <= i && i < index_t{N},
                      "Index ",
                      i,
                      " is not within bounds of a tensor of dimension ",
                      N);
  }
};

template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
class GenericPackedTensorAccessor
    : public GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST GenericPackedTensorAccessor(PtrType data_,
                                       const index_t* sizes_,
                                       const index_t* strides_)
      : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(
            data_, sizes_, strides_) {}

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <typename source_index_t,
            class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  C10_HOST GenericPackedTensorAccessor(PtrType data_,
                                       const source_index_t* sizes_,
                                       const source_index_t* strides_)
      : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(
            data_, sizes_, strides_) {}

  C10_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](
      index_t i) {
    index_t* new_sizes = this->sizes_ + 1;
    index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i, new_sizes, new_strides);
  }

  C10_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](
      index_t i) const {
    const index_t* new_sizes = this->sizes_ + 1;
    const index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i, new_sizes, new_strides);
  }

  /// Returns a PackedTensorAccessor of the same dimension after transposing the
  /// two dimensions given. Does not actually move elements; transposition is
  /// made by permuting the size/stride arrays. If the dimensions are not valid,
  /// asserts.
  C10_HOST GenericPackedTensorAccessor<T, N, PtrTraits, index_t> transpose(
      index_t dim1, index_t dim2) const {
    this->bounds_check_(dim1);
    this->bounds_check_(dim2);
    GenericPackedTensorAccessor<T, N, PtrTraits, index_t> result(
        this->data_, this->sizes_, this->strides_);
    std::swap(result.strides_[dim1], result.strides_[dim2]);
    std::swap(result.sizes_[dim1], result.sizes_[dim2]);
    return result;
  }
};

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
