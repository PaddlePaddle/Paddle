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

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/funcs/elementwise_base.h"

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/pten/kernels/hybird/eigen/common.h"

namespace pten {

// Add
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsAddFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsAddFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto blas = paddle::operators::math::GetBlas<DevCtx, T>(dev_ctx);
    blas.VADD(x.numel(), x.data<T>(), y.data<T>(), z->mutable_data<T>());
  }
};

template <typename DevCtx, typename T>
struct SameDimsAddFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    z->mutable_data<T>();
    auto eigen_x = pten::EigenVector<T>::Flatten(x);
    auto eigen_y = pten::EigenVector<T>::Flatten(y);
    auto eigen_z = pten::EigenVector<T>::Flatten(*z);
    auto& place = *dev_ctx.eigen_device();
    eigen_z.device(place) = eigen_x + eigen_y;
  }
};

// Subtract
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsSubtractFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsSubtractFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto blas = paddle::operators::math::GetBlas<DevCtx, T>(dev_ctx);
    blas.VSUB(x.numel(), x.data<T>(), y.data<T>(), z->mutable_data<T>());
  }
};

template <typename DevCtx, typename T>
struct SameDimsSubtractFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto eigen_x = pten::EigenVector<T>::Flatten(x);
    auto eigen_y = pten::EigenVector<T>::Flatten(y);
    auto eigen_z = pten::EigenVector<T>::Flatten(*z);
    auto& place = *dev_ctx.eigen_device();
    eigen_z.device(place) = eigen_x - eigen_y;
  }
};

// Divide
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsDivideFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsDivideFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    paddle::platform::errors::InvalidArgument(
        "If use SameDimsDivideFunctor, template args(T) must be floating "
        "point. ");
  }
};

template <typename DevCtx, typename T>
struct SameDimsDivideFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto blas = paddle::operators::math::GetBlas<DevCtx, T>(dev_ctx);
    blas.VDIV(x.numel(), x.data<T>(), y.data<T>(), z->mutable_data<T>());
  }
};

// Multiply
template <typename DevCtx, typename T, class Enable = void>
struct SameDimsMultiplyFunctor {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z);
};

template <typename DevCtx, typename T>
struct SameDimsMultiplyFunctor<
    DevCtx,
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto blas = paddle::operators::math::GetBlas<DevCtx, T>(dev_ctx);
    blas.VMUL(x.numel(), x.data<T>(), y.data<T>(), z->mutable_data<T>());
  }
};

template <typename DevCtx, typename T>
struct SameDimsMultiplyFunctor<
    DevCtx,
    T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const DevCtx& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    auto eigen_x = pten::EigenVector<T>::Flatten(x);
    auto eigen_y = pten::EigenVector<T>::Flatten(y);
    auto eigen_z = pten::EigenVector<T>::Flatten(*z);
    auto& place = *dev_ctx.eigen_device();
    eigen_z.device(place) = eigen_x * eigen_y;
  }
};

inline void UpdateElementwiseIndexArray(const int* out_dims_array,
                                        const int max_dim,
                                        int* index_array) {
  for (int i = max_dim - 1; i >= 0; --i) {
    ++index_array[i];
    if (index_array[i] >= out_dims_array[i]) {
      index_array[i] -= out_dims_array[i];
    } else {
      break;
    }
  }
}

inline int GetElementwiseIndex(const int* x_dims_array,
                               const int max_dim,
                               const int* index_array) {
  int index_ = 0;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] > 1) {
      index_ = index_ * x_dims_array[i] + index_array[i];
    }
  }
  return index_;
}

template <typename Functor, typename T, typename OutType = T>
void CommonForwardBroadcastCPU(const DenseTensor& x,
                               const DenseTensor& y,
                               DenseTensor* z,
                               int* x_dims_array,
                               int* y_dims_array,
                               int* out_dims_array,
                               int max_dim,
                               const paddle::platform::CPUDeviceContext& ctx,
                               Functor func,
                               const bool is_xsize_larger = true) {
  std::vector<int> index_array(max_dim, 0);
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  PADDLE_ENFORCE_NOT_NULL(x_data,
                          paddle::platform::errors::InvalidArgument(
                              "The input X should not be empty."));
  PADDLE_ENFORCE_NOT_NULL(y_data,
                          paddle::platform::errors::InvalidArgument(
                              "The input Y should not be empty."));
  OutType* out_data = z->mutable_data<OutType>();

  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = GetElementwiseIndex(x_dims_array, max_dim, index_array.data());
    y_index = GetElementwiseIndex(y_dims_array, max_dim, index_array.data());
    if (is_xsize_larger) {
      out_data[out_index] = func(x_data[x_index], y_data[y_index]);
    } else {
      out_data[out_index] = func(y_data[y_index], x_data[x_index]);
    }

    UpdateElementwiseIndexArray(out_dims_array, max_dim, index_array.data());
  }
}

template <typename Functor, typename T, typename OutType = T>
void CommonElementwiseBroadcastForward(
    const paddle::platform::CPUDeviceContext& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    DenseTensor* z,
    const DDim& x_dims,
    const DDim& y_dims,
    Functor func,
    int axis,
    const bool is_xsize_larger = true) {
  int max_dim = (std::max)(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
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
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  funcs::GetBroadcastDimsArrays(x_dims,
                                y_dims,
                                x_dims_array.data(),
                                y_dims_array.data(),
                                out_dims_array.data(),
                                max_dim,
                                axis);

  CommonForwardBroadcastCPU<Functor, T, OutType>(x,
                                                 y,
                                                 z,
                                                 x_dims_array.data(),
                                                 y_dims_array.data(),
                                                 out_dims_array.data(),
                                                 max_dim,
                                                 dev_ctx,
                                                 func,
                                                 is_xsize_larger);
}

// It is a common CPU implementation to compute binary calculation with the
// support of broadcast. Note:
// 1. CPU implementation cannot support the case when x needs broadcast, thus
//    this function need to be called with XxxFunctor and XxxInverseFunctor,
//    like AddFunctor and InverseAddFunctor.
// 2. The corresponding GPU implementation supports all the broadcast cases,
//    thus there is no need to define and call with XxxInverseFunctor.
// TODO(liuyiqun): optimize the CPU implementation to support all broadcast
// cases and avoid the need of XxxInverseFunctor.
template <typename Functor, typename T, typename OutType = T>
void ElementwiseCompute(const paddle::platform::CPUDeviceContext& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        int axis,
                        Functor func,
                        DenseTensor* z) {
  z->mutable_data<OutType>();
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  bool is_xsize_larger = true;
  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }
  funcs::
      TransformFunctor<Functor, T, paddle::platform::CPUDeviceContext, OutType>
          functor(x, y, z, dev_ctx, func, is_xsize_larger);
  if (x_dims == y_dims) {
    functor.Run();
    return;
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
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

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimed = funcs::trim_trailing_singular_dims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    funcs::get_mid_dims(x_dims,
                        y_dims_trimed,
                        axis_trim,
                        &pre,
                        &n,
                        &post,
                        &is_run_common_broadcast);
  } else {
    auto x_dims_trimed = funcs::trim_trailing_singular_dims(x_dims);
    axis_trim = (x_dims_trimed.size() == 0) ? y_dims.size() : axis;
    funcs::get_mid_dims(y_dims,
                        x_dims_trimed,
                        axis_trim,
                        &pre,
                        &n,
                        &post,
                        &is_run_common_broadcast);
  }
  // special case for common implementation.
  // case 1: x=[2,3,1,5], y=[2,1,4,1]
  // case 2: x=[2,3,4], y=[1,1,4]
  if (is_run_common_broadcast == 1) {
    CommonElementwiseBroadcastForward<Functor, T, OutType>(
        dev_ctx, x, y, z, x_dims, y_dims, func, axis, is_xsize_larger);
    return;
  }

  if (post == 1) {
    functor.RunRowWise(n, pre);
    return;
  } else {
    functor.RunMidWise(n, pre, post);
    return;
  }
}

template <typename Functor>
struct SameDimsElementwiseCompute {
  void operator()(const paddle::platform::CPUDeviceContext& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    Functor()(dev_ctx, x, y, z);
  }
};

}  // namespace pten
