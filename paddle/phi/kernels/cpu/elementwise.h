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

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/funcs/common_shape.h"
#include "paddle/pten/kernels/funcs/elementwise_base.h"

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/pten/kernels/funcs/eigen/common.h"

namespace pten {

// FORWARD CODE

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
    blas.VADD(
        x.numel(), x.data<T>(), y.data<T>(), dev_ctx.template Alloc<T>(z));
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
    dev_ctx.template Alloc<T>(z);
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
    blas.VSUB(
        x.numel(), x.data<T>(), y.data<T>(), dev_ctx.template Alloc<T>(z));
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
    blas.VDIV(
        x.numel(), x.data<T>(), y.data<T>(), dev_ctx.template Alloc<T>(z));
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
    blas.VMUL(
        x.numel(), x.data<T>(), y.data<T>(), dev_ctx.template Alloc<T>(z));
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

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonGradBroadcastCPU(const DenseTensor& x,
                            const DenseTensor& y,
                            const DenseTensor& out,
                            const DenseTensor& dout,
                            DenseTensor* dx,
                            DenseTensor* dy,
                            int* x_dims_array,
                            int* y_dims_array,
                            int* out_dims_array,
                            int max_dim,
                            const CPUContext& ctx,
                            DX_OP dx_op,
                            DY_OP dy_op) {
  std::vector<int> index_array(max_dim, 0);
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  const Tout* out_data = out.data<Tout>();
  const Tout* dout_data = dout.data<Tout>();
  T* dx_data = dx == nullptr ? nullptr : ctx.Alloc<T>(dx);
  T* dy_data = dy == nullptr ? nullptr : ctx.Alloc<T>(dy);
  if (dx_data != nullptr) {
    memset(dx_data, 0, dx->numel() * sizeof(T));
  }
  if (dy_data != nullptr) {
    memset(dy_data, 0, dy->numel() * sizeof(T));
  }
  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = GetElementwiseIndex(x_dims_array, max_dim, index_array.data());
    y_index = GetElementwiseIndex(y_dims_array, max_dim, index_array.data());
    if (dx_data != nullptr) {
      dx_data[x_index] += dx_op(x_data[x_index],
                                y_data[y_index],
                                out_data[out_index],
                                dout_data[out_index]);
    }
    if (dy_data != nullptr) {
      dy_data[y_index] += dy_op(x_data[x_index],
                                y_data[y_index],
                                out_data[out_index],
                                dout_data[out_index]);
    }

    UpdateElementwiseIndexArray(out_dims_array, max_dim, index_array.data());
  }
}

template <typename Functor, typename T, typename OutType = T>
void CommonForwardBroadcastCPU(const DenseTensor& x,
                               const DenseTensor& y,
                               DenseTensor* z,
                               int* x_dims_array,
                               int* y_dims_array,
                               int* out_dims_array,
                               int max_dim,
                               const CPUContext& ctx,
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
  OutType* out_data = ctx.Alloc<OutType>(z);

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
void CommonElementwiseBroadcastForward(const CPUContext& dev_ctx,
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
void ElementwiseCompute(const CPUContext& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        int axis,
                        Functor func,
                        DenseTensor* z) {
  dev_ctx.Alloc<OutType>(z);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  bool is_xsize_larger = true;
  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }
  funcs::TransformFunctor<Functor, T, CPUContext, OutType> functor(
      x, y, z, dev_ctx, func, is_xsize_larger);
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
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* z) {
    Functor()(dev_ctx, x, y, z);
  }
};

// BACKWARD CODE

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static void ElemwiseGradBroadcast1CPU(const T* x,
                                      const T* y,
                                      const Tout* out,
                                      const Tout* dout,
                                      int h,
                                      int w,
                                      bool is_xsize_larger,
                                      DX_OP dx_op,
                                      DY_OP dy_op,
                                      T* dx,
                                      T* dy) {
  if (is_xsize_larger) {
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        int x_offset = i * w + j;
        if (dx != nullptr) {
          dx[x_offset] =
              dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
        }
        if (dy != nullptr) {
          T tmp = dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
          if (i == 0) {
            dy[j] = tmp;
          } else {
            dy[j] += tmp;
          }
        }
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        int y_offset = i * w + j;
        if (dy != nullptr) {
          dy[y_offset] =
              dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
        }
        if (dx != nullptr) {
          T tmp = dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
          if (i == 0) {
            dx[j] = tmp;
          } else {
            dx[j] += tmp;
          }
        }
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static void ElemwiseGradBroadcast2CPU(const T* x,
                                      const T* y,
                                      const Tout* out,
                                      const Tout* dout,
                                      int pre,
                                      int n,
                                      int post,
                                      bool is_xsize_larger,
                                      DX_OP dx_op,
                                      DY_OP dy_op,
                                      T* dx,
                                      T* dy) {
  if (is_xsize_larger) {
    for (int i = 0; i < pre; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < post; ++k) {
          int x_offset = i * n * post + j * post + k;
          if (dx != nullptr) {
            dx[x_offset] =
                dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
          }
          if (dy != nullptr) {
            T tmp = dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
            if (i == 0 && k == 0) {
              dy[j] = tmp;
            } else {
              dy[j] += tmp;
            }
          }
        }
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    for (int i = 0; i < pre; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < post; ++k) {
          int y_offset = i * n * post + j * post + k;
          if (dy != nullptr) {
            dy[y_offset] =
                dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
          }
          if (dx != nullptr) {
            T tmp = dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
            if (i == 0 && k == 0) {
              dx[j] = tmp;
            } else {
              dx[j] += tmp;
            }
          }
        }
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonElementwiseBroadcastBackward(const CPUContext& ctx,
                                        const DDim& x_dims,
                                        const DDim& y_dims,
                                        const DenseTensor& x,
                                        const DenseTensor& y,
                                        const DenseTensor& out,
                                        const DenseTensor& dout,
                                        int axis,
                                        DenseTensor* dx,
                                        DenseTensor* dy,
                                        DX_OP dx_op,
                                        DY_OP dy_op) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
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
  // for inplace strategy. memset will make dx and dout clear and get wrong
  // result.
  if (dx && dx->IsSharedBufferWith(dout)) {
    dx->clear();
    dx->mutable_data<T>(x_dims, ctx.GetPlace());
  }

  VLOG(3) << "CommonElementwiseBroadcastBackward xdims:"
          << pten::framework::make_ddim(x_dims_array)
          << " ydim:" << pten::framework::make_ddim(y_dims_array);

  CommonGradBroadcastCPU<T, DX_OP, DY_OP, Tout>(x,
                                                y,
                                                out,
                                                dout,
                                                dx,
                                                dy,
                                                x_dims_array.data(),
                                                y_dims_array.data(),
                                                out_dims_array.data(),
                                                max_dim,
                                                ctx,
                                                dx_op,
                                                dy_op);
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void ElemwiseGradComputeWithBroadcast(const CPUContext& ctx,
                                      const DDim& x_dims,
                                      const DDim& y_dims,
                                      const DenseTensor& x,
                                      const DenseTensor& y,
                                      const DenseTensor& out,
                                      const DenseTensor& dout,
                                      int axis,
                                      DenseTensor* dx,
                                      DenseTensor* dy,
                                      DX_OP dx_op,
                                      DY_OP dy_op) {
  bool is_xsize_larger = true;

  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
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
  // special case for common backward implementation.
  if (is_run_common_broadcast) {
    CommonElementwiseBroadcastBackward<T, DX_OP, DY_OP, Tout>(
        ctx, x_dims, y_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
    return;
  }
  if (post == 1) {
    ElemwiseGradBroadcast1CPU(x.data<T>(),
                              y.data<T>(),
                              out.data<Tout>(),
                              dout.data<Tout>(),
                              pre,
                              n,
                              is_xsize_larger,
                              dx_op,
                              dy_op,
                              dx == nullptr ? nullptr : ctx.Alloc<T>(dx),
                              dy == nullptr ? nullptr : ctx.Alloc<T>(dy));
  } else {
    ElemwiseGradBroadcast2CPU(x.data<T>(),
                              y.data<T>(),
                              out.data<Tout>(),
                              dout.data<Tout>(),
                              pre,
                              n,
                              post,
                              is_xsize_larger,
                              dx_op,
                              dy_op,
                              dx == nullptr ? nullptr : ctx.Alloc<T>(dx),
                              dy == nullptr ? nullptr : ctx.Alloc<T>(dy));
  }
}

// NOTE(dzhwinter): Only used in elementwise_add, elementwise_sub.
// explicit gradient can cut off X, Y, Out from gradient op
// In elementwise_add, elementwise_sub, we use dout as fake X, Y, Out to reuse
// elementwise code.
template <typename T, typename DX_OP, typename DY_OP>
void ElemwiseExplicitGradCompute(const CPUContext& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 const DenseTensor& out,
                                 const DenseTensor& dout,
                                 int axis,
                                 DenseTensor* dx,
                                 DenseTensor* dy,
                                 DX_OP dx_op,
                                 DY_OP dy_op) {
  const DDim& x_dim = x.dims();
  const DDim& y_dim = y.dims();
  if (x.dims() == y.dims()) {
    pten::funcs::ElemwiseGradComputeNoBroadcast<CPUContext, T, DX_OP, DY_OP>(
        dev_ctx,
        x_dim,
        y_dim,
        dout,
        dout,
        out,
        dout,
        axis,
        dx,
        dy,
        dx_op,
        dy_op);
  } else {
    ElemwiseGradComputeWithBroadcast<T, DX_OP, DY_OP>(dev_ctx,
                                                      x_dim,
                                                      y_dim,
                                                      dout,
                                                      dout,
                                                      out,
                                                      dout,
                                                      axis,
                                                      dx,
                                                      dy,
                                                      dx_op,
                                                      dy_op);
  }
}

/*
******************************
    Add Grad
******************************
*/
template <typename T>
struct IdentityGrad {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value>::type
elementwise_add_grad(const CPUContext& ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out,
                     const DenseTensor& dout,
                     DenseTensor* dx,
                     DenseTensor* dy,
                     int axis = -1) {
  auto blas = paddle::operators::math::GetBlas<CPUContext, T>(ctx);
  if (dx) {
    blas.VCOPY(
        dout.numel(), dout.data<T>(), dx->mutable_data<T>(ctx.GetPlace()));
  }

  if (dy) {
    blas.VCOPY(
        dout.numel(), dout.data<T>(), dy->mutable_data<T>(ctx.GetPlace()));
  }
}

template <typename T>
typename std::enable_if<!std::is_floating_point<T>::value>::type
elementwise_add_grad(const CPUContext& ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& out,
                     const DenseTensor& dout,
                     DenseTensor* dx,
                     DenseTensor* dy,
                     int axis = -1) {
  ElemwiseExplicitGradCompute<T, IdentityGrad<T>, IdentityGrad<T>>(
      ctx, x, y, out, dout, axis, dx, dy, IdentityGrad<T>(), IdentityGrad<T>());
}

/*
******************************
    Sub Grad
******************************
*/

template <typename T>
struct SubGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename T>
struct SubGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return -dout; }
};

template <typename T>
void elementwise_sub_grad(const CPUContext& ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          const DenseTensor& out,
                          const DenseTensor& dout,
                          DenseTensor* dx,
                          DenseTensor* dy,
                          int axis = -1) {
  ElemwiseExplicitGradCompute<T, SubGradDX<T>, SubGradDY<T>>(
      ctx, x, y, out, dout, axis, dx, dy, SubGradDX<T>(), SubGradDY<T>());
}

}  // namespace pten
