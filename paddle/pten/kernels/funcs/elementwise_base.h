/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/platform/transform.h"
#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/core/dense_tensor.h"

namespace pten {
namespace funcs {

using DDim = paddle::framework::DDim;

template <typename T, typename DeviceContext>
class RowwiseTransformIterator;

template <typename T, typename DeviceContext>
class MidWiseTransformIterator;

// NOTE(dzhwinter): ptrdiff_t in iterator is deperecated in c++17
template <typename T>
class RowwiseTransformIterator<T, CPUContext>
    : public std::iterator<std::random_access_iterator_tag,
                           T,
                           std::ptrdiff_t,
                           T *,
                           T &> {
 public:
  RowwiseTransformIterator(const T *ptr, int n) : ptr_(ptr), i_(0), n_(n) {}

  RowwiseTransformIterator<T, CPUContext> &operator++() {
    ++i_;
    if (UNLIKELY(i_ == n_)) {
      i_ = 0;
    }
    return *this;
  }

  RowwiseTransformIterator<T, CPUContext> &operator+(int n) {
    while (n-- > 0) {
      ++i_;
      if (UNLIKELY(i_ == n_)) {
        i_ = 0;
      }
    }

    return *this;
  }

  bool operator==(const RowwiseTransformIterator<T, CPUContext> &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(const RowwiseTransformIterator<T, CPUContext> &rhs) const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T &operator*() { return ptr_[i_]; }

 private:
  const T *ptr_;
  int i_;
  int64_t n_;
};

template <typename T>
class MidWiseTransformIterator<T, CPUContext>
    : public std::iterator<std::random_access_iterator_tag,
                           T,
                           std::ptrdiff_t,
                           T *,
                           T &> {
 public:
  MidWiseTransformIterator(const T *ptr, int n, int post)
      : ptr_(ptr), i_(0), j_(0), n_(n), post_(post) {}

  MidWiseTransformIterator<T, CPUContext> &operator++() {
    ++j_;
    if (UNLIKELY(j_ == post_)) {
      ++i_;
      j_ = 0;
      if (UNLIKELY(i_ == n_)) {
        i_ = 0;
      }
    }
    return *this;
  }

  MidWiseTransformIterator<T, CPUContext> &operator+(int n) {
    while (n-- > 0) {
      ++j_;
      if (UNLIKELY(j_ == post_)) {
        ++i_;
        j_ = 0;
        if (UNLIKELY(i_ == n_)) {
          i_ = 0;
        }
      }
    }
    return *this;
  }

  bool operator==(const MidWiseTransformIterator<T, CPUContext> &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(const MidWiseTransformIterator<T, CPUContext> &rhs) const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T &operator*() { return ptr_[i_]; }

 private:
  const T *ptr_;
  int64_t i_;
  int64_t j_;
  int64_t n_;
  int64_t post_;
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
class RowwiseTransformIterator<T, GPUContext>
    : public thrust::iterator_adaptor<RowwiseTransformIterator<T, GPUContext>,
                                      const T *> {
 public:
  typedef thrust::iterator_adaptor<RowwiseTransformIterator<T, GPUContext>,
                                   const T *>
      super_t;
  HOSTDEVICE RowwiseTransformIterator(const T *x, int n)
      : super_t(x), begin_(x), n_(n) {}
  friend class thrust::iterator_core_access;

 private:
  unsigned int n_;
  const T *begin_;
  HOSTDEVICE typename super_t::reference dereference() const {
    return *(begin_ + (this->base() - begin_) % n_);
  }
};

template <typename T>
class MidWiseTransformIterator<T, GPUContext>
    : public thrust::iterator_adaptor<MidWiseTransformIterator<T, GPUContext>,
                                      const T *> {
 public:
  typedef thrust::iterator_adaptor<MidWiseTransformIterator<T, GPUContext>,
                                   const T *>
      super_t;
  HOSTDEVICE MidWiseTransformIterator(const T *x, int n, int post)
      : super_t(x), begin_(x), n_(n), post_(post) {}
  friend class thrust::iterator_core_access;

 private:
  unsigned int post_;
  unsigned int n_;
  const T *begin_;
  HOSTDEVICE typename super_t::reference dereference() const {
    return *(begin_ + (((this->base() - begin_) / post_) % n_));
  }
};
#endif

template <typename Functor,
          typename T,
          typename DeviceContext,
          typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const DenseTensor &x,
                   const DenseTensor &y,
                   DenseTensor *z,
                   const DeviceContext &ctx,
                   Functor func,
                   const bool is_xsize_larger = true)
      : x_(x.data<T>()),
        y_(y.data<T>()),
        z_(z->mutable_data<OutType>()),
        nx_(x.numel()),
        ctx_(ctx),
        func_(func),
        is_xsize_larger_(is_xsize_larger) {
    if (is_xsize_larger_ == false) {
      nx_ = y.numel();
    }
  }

  inline void Run() const {
    paddle::platform::Transform<DeviceContext> trans;
    trans(ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n, int pre) const {
    paddle::platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_,
            x_,
            x_ + nx_,
            RowwiseTransformIterator<T, DeviceContext>(y_, n),
            z_,
            func_);
    } else {
      trans(ctx_,
            y_,
            y_ + nx_,
            RowwiseTransformIterator<T, DeviceContext>(x_, n),
            z_,
            func_);
    }
  }

  inline void RunMidWise(int n, int pre, int post) const {
    paddle::platform::Transform<DeviceContext> trans;
    if (is_xsize_larger_) {
      trans(ctx_,
            x_,
            x_ + nx_,
            MidWiseTransformIterator<T, DeviceContext>(y_, n, post),
            z_,
            func_);
    } else {
      trans(ctx_,
            y_,
            y_ + nx_,
            MidWiseTransformIterator<T, DeviceContext>(x_, n, post),
            z_,
            func_);
    }
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const DeviceContext &ctx_;
  Functor func_;
  bool is_xsize_larger_;
};

inline DDim trim_trailing_singular_dims(const DDim &dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims.size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if (dims[actual_dims_size - 1] != 1) break;
  }
  if (actual_dims_size == dims.size()) return dims;
  std::vector<int> trim_dims;
  trim_dims.resize(actual_dims_size);
  for (int i = 0; i < actual_dims_size; ++i) {
    trim_dims[i] = dims[i];
  }
  if (trim_dims.size() == 0) {
    return DDim(paddle::framework::make_dim());
  }
  DDim actual_dims = paddle::framework::make_ddim(trim_dims);
  return actual_dims;
}

/*
 * Out = X ⊙ Y
 * If Y's shape does not match X' shape, they will be reshaped.
 * For example:
 * 1. shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
 *    pre=2, n=3*4, post=5
 *    x.shape(2, 12, 5) * y.shape(1, 12, 1).broadcast(2, 12, 5)
 * 2. shape(X) = (2, 3, 4, 5), shape(Y) = (4,5)
 *    pre=2*3, n=4*5, post=1
 *    x.shape(6, 20, 1) * y.shape(1, 20, 1).broadcast(6, 20, 1)
 *
 * New parameter: *is_run_common_broadcast* is a flag to record whether to run
 * common broadcast code.
 */
inline void get_mid_dims(const DDim &x_dims,
                         const DDim &y_dims,
                         const int axis,
                         int *pre,
                         int *n,
                         int *post,
                         int *is_run_common_broadcast) {
  *pre = 1;
  *n = 1;
  *post = 1;
  *is_run_common_broadcast = 0;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    if (x_dims[i + axis] != y_dims[i]) {
      PADDLE_ENFORCE_EQ(y_dims[i] == 1 || x_dims[i + axis] == 1,
                        true,
                        paddle::platform::errors::InvalidArgument(
                            "Broadcast dimension mismatch. Operands "
                            "could not be broadcast together with the shape of "
                            "X = [%s] and the shape of Y = [%s]. Received [%d] "
                            "in X is not equal to [%d] in Y.",
                            x_dims,
                            y_dims,
                            x_dims[i + axis],
                            y_dims[i]));
      *is_run_common_broadcast = 1;
      return;
    }
    (*n) *= y_dims[i];
  }
  for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
    (*post) *= x_dims[i];
  }
}

inline void GetBroadcastDimsArrays(const DDim &x_dims,
                                   const DDim &y_dims,
                                   int *x_dims_array,
                                   int *y_dims_array,
                                   int *out_dims_array,
                                   const int max_dim,
                                   const int axis) {
  PADDLE_ENFORCE_GE(
      axis,
      0,
      paddle::platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    paddle::platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));
  if (x_dims.size() > y_dims.size()) {
    std::fill(y_dims_array, y_dims_array + axis, 1);
    if (axis + y_dims.size() < max_dim) {
      std::fill(y_dims_array + axis + y_dims.size(), y_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array + axis);
  } else {
    std::fill(x_dims_array, x_dims_array + axis, 1);
    if (axis + x_dims.size() < max_dim) {
      std::fill(x_dims_array + axis + x_dims.size(), x_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array + axis);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array);
  }

  for (int i = 0; i < max_dim; i++) {
    PADDLE_ENFORCE_EQ(
        x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 ||
            y_dims_array[i] <= 1,
        true,
        paddle::platform::errors::InvalidArgument(
            "Broadcast dimension mismatch. Operands could "
            "not be broadcast together with the shape of X = [%s] and "
            "the shape of Y = [%s]. Received [%d] in X is not equal to "
            "[%d] in Y at i:%d.",
            x_dims,
            y_dims,
            x_dims_array[i],
            y_dims_array[i],
            i));
    if ((x_dims_array[i] > 1 || y_dims_array[i] > 1) ||
        (x_dims_array[i] == 1 && y_dims_array[i] == 1)) {
      out_dims_array[i] = (std::max)(x_dims_array[i], y_dims_array[i]);
    } else {
      out_dims_array[i] = -1;
    }
  }
}
}  // namespace funcs
}  // namespace pten
