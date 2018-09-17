/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/transform.h"

#ifdef __NVCC__
#include <cuda.h>
#include <thrust/iterator/iterator_adaptor.h>
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
constexpr int ELEMWISE_MAX_BLOCK_DIM = 1024;
#endif

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

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
 */
inline void get_mid_dims(const framework::DDim &x_dims,
                         const framework::DDim &y_dims, const int axis,
                         int *pre, int *n, int *post) {
  *pre = 1;
  *n = 1;
  *post = 1;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_dims[i];
  }

  for (int i = 0; i < y_dims.size(); ++i) {
    PADDLE_ENFORCE_EQ(x_dims[i + axis], y_dims[i],
                      "Broadcast dimension mismatch.");
    (*n) *= y_dims[i];
  }

  for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
    (*post) *= x_dims[i];
  }
}

inline framework::DDim trim_trailing_singular_dims(
    const framework::DDim &dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims.size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if (dims[actual_dims_size - 1] != 1) break;
  }

  std::vector<int> trim_dims;
  trim_dims.resize(actual_dims_size);
  for (int i = 0; i < actual_dims_size; ++i) {
    trim_dims[i] = dims[i];
  }
  if (trim_dims.size() == 0) {
    return framework::DDim(framework::make_dim());
  }
  framework::DDim actual_dims = framework::make_ddim(trim_dims);
  return actual_dims;
}

template <typename T, typename DeviceContext>
class RowwiseTransformIterator;

template <typename T, typename DeviceContext>
class MidWiseTransformIterator;

// NOTE(dzhwinter): ptrdiff_t in iterator is deperecated in c++17
template <typename T>
class RowwiseTransformIterator<T, platform::CPUDeviceContext>
    : public std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t,
                           T *, T &> {
 public:
  RowwiseTransformIterator(const T *ptr, int n) : ptr_(ptr), i_(0), n_(n) {}

  RowwiseTransformIterator<T, platform::CPUDeviceContext> &operator++() {
    ++i_;
    if (UNLIKELY(i_ == n_)) {
      i_ = 0;
    }
    return *this;
  }

  bool operator==(const RowwiseTransformIterator<T, platform::CPUDeviceContext>
                      &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(const RowwiseTransformIterator<T, platform::CPUDeviceContext>
                      &rhs) const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T &operator*() { return ptr_[i_]; }

 private:
  const T *ptr_;
  int i_;
  int64_t n_;
};

template <typename T>
class MidWiseTransformIterator<T, platform::CPUDeviceContext>
    : public std::iterator<std::random_access_iterator_tag, T, std::ptrdiff_t,
                           T *, T &> {
 public:
  MidWiseTransformIterator(const T *ptr, int n, int post)
      : ptr_(ptr), i_(0), j_(0), n_(n), post_(post) {}

  MidWiseTransformIterator<T, platform::CPUDeviceContext> &operator++() {
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

  bool operator==(const MidWiseTransformIterator<T, platform::CPUDeviceContext>
                      &rhs) const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(const MidWiseTransformIterator<T, platform::CPUDeviceContext>
                      &rhs) const {
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

#ifdef __NVCC__
template <typename T>
class RowwiseTransformIterator<T, platform::CUDADeviceContext>
    : public thrust::iterator_adaptor<
          RowwiseTransformIterator<T, platform::CUDADeviceContext>, const T *> {
 public:
  typedef thrust::iterator_adaptor<
      RowwiseTransformIterator<T, platform::CUDADeviceContext>, const T *>
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
class MidWiseTransformIterator<T, platform::CUDADeviceContext>
    : public thrust::iterator_adaptor<
          MidWiseTransformIterator<T, platform::CUDADeviceContext>, const T *> {
 public:
  typedef thrust::iterator_adaptor<
      MidWiseTransformIterator<T, platform::CUDADeviceContext>, const T *>
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

template <typename Functor, typename T, typename DeviceContext,
          typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const framework::Tensor *x, const framework::Tensor *y,
                   framework::Tensor *z, const DeviceContext &ctx, Functor func)
      : x_(x->data<T>()),
        y_(y->data<T>()),
        z_(z->mutable_data<OutType>(ctx.GetPlace())),
        nx_(x->numel()),
        ctx_(ctx),
        func_(func) {}

  inline void Run() const {
    platform::Transform<DeviceContext> trans;
    trans(ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n, int pre) const {
    platform::Transform<DeviceContext> trans;
    trans(ctx_, x_, x_ + nx_, RowwiseTransformIterator<T, DeviceContext>(y_, n),
          z_, func_);
  }

  inline void RunMidWise(int n, int pre, int post) const {
    platform::Transform<DeviceContext> trans;
    trans(ctx_, x_, x_ + nx_,
          MidWiseTransformIterator<T, DeviceContext>(y_, n, post), z_, func_);
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const DeviceContext &ctx_;
  Functor func_;
};

#define EIGEN_FUNCTOR(name, eigen_op)                                          \
  struct Eigen##name##Functor {                                                \
    template <typename DeviceContext, typename T>                              \
    inline void Run(const framework::Tensor *x, const framework::Tensor *y,    \
                    framework::Tensor *z,                                      \
                    const framework::ExecutionContext &ctx) {                  \
      auto x_e = framework::EigenVector<T>::Flatten(*x);                       \
      auto y_e = framework::EigenVector<T>::Flatten(*y);                       \
      auto z_e = framework::EigenVector<T>::Flatten(*z);                       \
      z_e.device(                                                              \
          *ctx.template device_context<DeviceContext>().eigen_device()) =      \
          eigen_op(x_e, y_e);                                                  \
    }                                                                          \
    template <typename DeviceContext, typename T>                              \
    inline void RunBroadCast(const framework::Tensor *x,                       \
                             const framework::Tensor *y, framework::Tensor *z, \
                             const framework::ExecutionContext &ctx, int pre,  \
                             int n) {                                          \
      auto x_e = framework::EigenVector<T>::Flatten(*x);                       \
      auto y_e = framework::EigenVector<T>::Flatten(*y);                       \
      auto z_e = framework::EigenVector<T>::Flatten(*z);                       \
      auto y_bcast = y_e.reshape(Eigen::DSizes<int, 2>(1, n))                  \
                         .broadcast(Eigen::DSizes<int, 2>(pre, 1))             \
                         .reshape(Eigen::DSizes<int, 1>(x_e.size()));          \
      z_e.device(                                                              \
          *ctx.template device_context<DeviceContext>().eigen_device()) =      \
          eigen_op(x_e, y_bcast);                                              \
    }                                                                          \
    template <typename DeviceContext, typename T>                              \
    inline void RunBroadCast2(const framework::Tensor *x,                      \
                              const framework::Tensor *y,                      \
                              framework::Tensor *z,                            \
                              const framework::ExecutionContext &ctx, int pre, \
                              int n, int post) {                               \
      auto x_e = framework::EigenVector<T>::Flatten(*x);                       \
      auto y_e = framework::EigenVector<T>::Flatten(*y);                       \
      auto z_e = framework::EigenVector<T>::Flatten(*z);                       \
      auto y_bcast = y_e.reshape(Eigen::DSizes<int, 3>(1, n, 1))               \
                         .broadcast(Eigen::DSizes<int, 3>(pre, 1, post))       \
                         .reshape(Eigen::DSizes<int, 1>(x_e.size()));          \
      z_e.device(                                                              \
          *ctx.template device_context<DeviceContext>().eigen_device()) =      \
          eigen_op(x_e, y_bcast);                                              \
    }                                                                          \
  }

#define EIGEN_ADD(x, y) ((x) + (y))

EIGEN_FUNCTOR(Add, EIGEN_ADD);

#define EIGEN_SUB(x, y) ((x) - (y))

EIGEN_FUNCTOR(Sub, EIGEN_SUB);

#define EIGEN_MUL(x, y) ((x) * (y))

EIGEN_FUNCTOR(Mul, EIGEN_MUL);

#define EIGEN_DIV(x, y) ((x) / (y))

EIGEN_FUNCTOR(Div, EIGEN_DIV);

template <typename T, typename DX_OP, typename DY_OP>
struct ElemwiseGradNoBroadcast {
  const T *x_;
  const T *y_;
  const T *out_;
  const T *dout_;

  HOSTDEVICE void operator()(size_t i) {
    if (dx_ != nullptr) {
      dx_[i] = dx_op_(x_[i], y_[i], out_[i], dout_[i]);
    }
    if (dy_ != nullptr) {
      dy_[i] = dy_op_(x_[i], y_[i], out_[i], dout_[i]);
    }
  }

  DX_OP dx_op_;
  DY_OP dy_op_;
  T *dx_;
  T *dy_;
};

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast1CPU(const T *x, const T *y, const T *out,
                                      const T *dout, int h, int w, DX_OP dx_op,
                                      DY_OP dy_op, T *dx, T *dy) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int x_offset = i * w + j;
      if (dx != nullptr) {
        dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
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
}

#ifdef __NVCC__
template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast1CUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int h, int w,
    DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  int j = blockIdx.x;
  int i = threadIdx.x;
  int tid = threadIdx.x;
  T val = 0;

  do {
    int x_offset = i * w + j;
    if (dx) {
      dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
    }
    if (dy) {
      val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
    }
    i += ELEMWISE_MAX_BLOCK_DIM;
  } while (i < h);

  if (dy) {
    h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
    val = paddle::platform::reduceSum(val, tid, h);
    if (threadIdx.x == 0) {
      dy[j] = val;
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast1CUDA(cudaStream_t stream, const T *x,
                                       const T *y, const T *out, const T *dout,
                                       int h, int w, DX_OP dx_op, DY_OP dy_op,
                                       T *dx, T *dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
  int gird_size = w;
  ElemwiseGradBroadcast1CUDAKernel<<<gird_size, block_size, 0, stream>>>(
      x, y, out, dout, h, w, dx_op, dy_op, dx, dy);
}

#endif

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast2CPU(const T *x, const T *y, const T *out,
                                      const T *dout, int pre, int n, int post,
                                      DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
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
}

#ifdef __NVCC__
template <typename T, typename DX_OP, typename DY_OP>
static __global__ void ElemwiseGradBroadcast2CUDAKernel(
    const T *x, const T *y, const T *out, const T *dout, int pre, int n,
    int post, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  T val = 0;
  int ttid = tid;

  while (true) {
    int i = ttid / post;
    int k = ttid % post;
    if (i >= pre) break;

    int x_offset = i * n * post + j * post + k;

    if (dx != nullptr) {
      dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
    }

    if (dy != nullptr) {
      val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
    }

    ttid += ELEMWISE_MAX_BLOCK_DIM;
  }

  if (dy) {
    int h = pre * post;
    h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
    val = paddle::platform::reduceSum(val, tid, h);
    if (threadIdx.x == 0) {
      dy[j] = val;
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast2CUDA(cudaStream_t stream, const T *x,
                                       const T *y, const T *out, const T *dout,
                                       int pre, int n, int post, DX_OP dx_op,
                                       DY_OP dy_op, T *dx, T *dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;
  ElemwiseGradBroadcast2CUDAKernel<<<gird_size, block_size, 0, stream>>>(
      x, y, out, dout, pre, n, post, dx_op, dy_op, dx, dy);
}

#endif

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseGradComputeNoBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim, const framework::Tensor &x,
    const framework::Tensor &y, const framework::Tensor &out,
    const framework::Tensor &dout, int axis, framework::Tensor *dx,
    framework::Tensor *dy, DX_OP dx_op, DY_OP dy_op) {
  size_t N = static_cast<size_t>(framework::product(x_dim));
#if !defined(_WIN32)
  platform::ForRange<DeviceContext> for_range(
      ctx.template device_context<DeviceContext>(), N);
#else
  platform::ForRange<DeviceContext> for_range(
      ctx.device_context<DeviceContext>(), N);
#endif  // !_WIN32
  for_range(ElemwiseGradNoBroadcast<T, DX_OP, DY_OP>{
      x.data<T>(), y.data<T>(), out.data<T>(), dout.data<T>(), dx_op, dy_op,
      dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
      dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace())});
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseGradComputeWithBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim_untrimed, const framework::Tensor &x,
    const framework::Tensor &y, const framework::Tensor &out,
    const framework::Tensor &dout, int axis, framework::Tensor *dx,
    framework::Tensor *dy, DX_OP dx_op, DY_OP dy_op) {
  axis = (axis == -1 ? x_dim.size() - y_dim_untrimed.size() : axis);
  auto y_dim = trim_trailing_singular_dims(y_dim_untrimed);
  axis = (y_dim.size() == 0) ? x_dim.size() : axis;

  int pre, n, post;
  get_mid_dims(x_dim, y_dim, axis, &pre, &n, &post);
  if (post == 1) {
    int h = pre;
    int w = n;
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
      ElemwiseGradBroadcast1CUDA(
          ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
          y.data<T>(), out.data<T>(), dout.data<T>(), h, w, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      ElemwiseGradBroadcast1CPU(
          x.data<T>(), y.data<T>(), out.data<T>(), dout.data<T>(), h, w, dx_op,
          dy_op, dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
    }
  } else {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
      ElemwiseGradBroadcast2CUDA(
          ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
          y.data<T>(), out.data<T>(), dout.data<T>(), pre, n, post, dx_op,
          dy_op, dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      ElemwiseGradBroadcast2CPU(
          x.data<T>(), y.data<T>(), out.data<T>(), dout.data<T>(), pre, n, post,
          dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
    }
  }
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseGradCompute(const framework::ExecutionContext &ctx,
                         const framework::Tensor &x, const framework::Tensor &y,
                         const framework::Tensor &out,
                         const framework::Tensor &dout, int axis,
                         framework::Tensor *dx, framework::Tensor *dy,
                         DX_OP dx_op, DY_OP dy_op) {
  const framework::DDim &x_dim = x.dims();
  const framework::DDim &y_dim = y.dims();
  if (x.dims() == y.dims()) {
    ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dim, y_dim, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
  } else {  // Y is a scalar
    ElemwiseGradComputeWithBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dim, y_dim, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
  }
}

// NOTE(dzhwinter): Only used in elementwise_add, elementwise_sub.
// explicit gradient can cut off X, Y, Out from gradient op
// In elementwise_add, elementwise_sub, we use dout as fake X, Y, Out to reuse
// elementwise code.
template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseExplicitGradCompute(const framework::ExecutionContext &ctx,
                                 const framework::Tensor &x,
                                 const framework::Tensor &y,
                                 const framework::Tensor &out,
                                 const framework::Tensor &dout, int axis,
                                 framework::Tensor *dx, framework::Tensor *dy,
                                 DX_OP dx_op, DY_OP dy_op) {
  if (dy == nullptr) {
    const framework::DDim &dx_dims = dout.dims();
    auto dy_dims = dx_dims;
    ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        ctx, dx_dims, dy_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
  } else {
    if (dout.dims() == dy->dims()) {
      const framework::DDim &dx_dims = dout.dims();
      const framework::DDim &dy_dims = dy->dims();
      ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP>(
          ctx, dx_dims, dy_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
    } else {  // Y is a scalar
      auto dx_dims = dout.dims();
      const framework::DDim &dy_dims = dy->dims();
      ElemwiseGradComputeWithBroadcast<DeviceContext, T, DX_OP, DY_OP>(
          ctx, dx_dims, dy_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
    }
  }
}

// Deprecated
template <typename DeviceContext, typename T, typename functor,
          typename broadcastfunctor, typename broadcast2functor>
void ElementwiseGradCompute(const framework::ExecutionContext &ctx,
                            const framework::Tensor *x,
                            const framework::Tensor *y,
                            const framework::Tensor *out,
                            const framework::Tensor *dout, int axis,
                            framework::Tensor *dx, framework::Tensor *dy) {
  auto &place = *ctx.template device_context<DeviceContext>().eigen_device();

  auto x_dims = x->dims();
  auto y_dims = y->dims();

  if (dx) {
    dx->mutable_data<T>(ctx.GetPlace());
  }
  if (dy) {
    dy->mutable_data<T>(ctx.GetPlace());
  }

  if (x_dims == y_dims) {
    functor f;
    f(place, x, y, out, dx, dy, dout);
    return;
  }

  axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
  trim_trailing_singular_dims(y_dims);
  axis = (y_dims.size() == 0) ? x_dims.size() : axis;

  int pre, n, post;
  get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

  if (post == 1) {
    broadcastfunctor f;
    f(place, x, y, out, dx, dy, dout, pre, n);
    return;
  } else {
    broadcast2functor f;
    f(place, x, y, out, dx, dy, dout, pre, n, post);
    return;
  }
}

template <typename Functor, typename DeviceContext, typename T,
          typename OutType = T>

void ElementwiseComputeEx(const framework::ExecutionContext &ctx,
                          const framework::Tensor *x,
                          const framework::Tensor *y, int axis, Functor func,
                          framework::Tensor *z) {
  TransformFunctor<Functor, T, DeviceContext, OutType> functor(
      x, y, z, ctx.template device_context<DeviceContext>(), func);
  auto x_dims = x->dims();
  auto y_dims_untrimed = y->dims();
  PADDLE_ENFORCE_GE(x_dims.size(), y_dims_untrimed.size(),
                    "Rank of first input must >= rank of second input.");

  if (x_dims == y_dims_untrimed) {
    functor.Run();
    return;
  }

  axis = (axis == -1 ? x_dims.size() - y_dims_untrimed.size() : axis);
  PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                 "Axis should be in range [0, x_dims)");
  auto y_dims = trim_trailing_singular_dims(y_dims_untrimed);
  axis = (y_dims.size() == 0) ? x_dims.size() : axis;

  int pre, n, post;
  get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);
  if (post == 1) {
    functor.RunRowWise(n, pre);
    return;
  } else {
    functor.RunMidWise(n, pre, post);
    return;
  }
}

// FusedElemwiseAndAct
// --- forward
template <typename T, typename CompoundFunctor, bool KeepIntermediateOut>
struct FusedElemwiseAndActNoBroadcast {
  HOSTDEVICE void operator()(size_t i) {
    T y_val = y_[i];
    T x_val = x_[i];
    if (KeepIntermediateOut) {
      T intermeidiate_out = compound_functor_.GetIntermediateOut(x_val, y_val);
      intermediate_out_[i] = intermeidiate_out;
      out_[i] =
          compound_functor_.GetOutUseIntermediateOut(x_val, intermeidiate_out);
    } else {
      out_[i] = compound_functor_.GetOut(x_val, y_val);
    }
  }

  const T *x_;
  const T *y_;
  CompoundFunctor compound_functor_;
  T *out_;
  T *intermediate_out_;
};

// FusedElemwiseAndActBroadcast1:
// In this case, X and Y can be reshaped to a matrix.
// For example shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5) and axis = -1 or 2,
// X can be reshaped to (6, 20) and Y can be reshaped to (1, 20)
template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast1CPU(const T *x, const T *y,
                                             CompoundFunctor compound_functor,
                                             int h, int w, T *out,
                                             T *intermediate_out) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int offset = i * w + j;

      T y_val = BcastY ? y[j] : y[offset];
      T x_val = BcastY ? x[offset] : x[j];
      int64_t intermediate_out_offset;
      if (KeepIntermediateOut) {
        T intermeidiate_out = compound_functor.GetIntermediateOut(x_val, y_val);

        if (SameShapeOfIntermediateOutAndOut) {
          // for the case of f1(f2(x, y))
          intermediate_out_offset = offset;
        } else if (BcastY) {
          intermediate_out_offset = j;
        } else {
          intermediate_out_offset = offset;
        }

        intermediate_out[intermediate_out_offset] = intermeidiate_out;
        out[offset] =
            compound_functor.GetOutUseIntermediateOut(x_val, intermeidiate_out);
      } else {
        out[offset] = compound_functor.GetOut(x_val, y_val);
      }
    }
  }
}

// FusedElemwiseAndActBroadcast2
// In this case, X and Y can be reshaped to a matrix.
// For example shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4) and axis = 1,
// X can be reshaped to (2, 12, 5) and Y can be reshaped to (1, 12, 1)
// pre = 2, n = 12, post = 5
template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast2CPU(const T *x, const T *y, int pre,
                                             int n, int post,
                                             CompoundFunctor compound_functor,
                                             T *out, T *intermediate_out) {
  for (int i = 0; i < pre; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < post; ++k) {
        int offset = i * n * post + j * post + k;

        T y_val = BcastY ? y[j] : y[offset];
        T x_val = BcastY ? x[offset] : x[j];
        int64_t intermediate_out_offset;

        if (KeepIntermediateOut) {
          T intermeidiate_out =
              compound_functor.GetIntermediateOut(x_val, y_val);

          if (SameShapeOfIntermediateOutAndOut) {
            // for the case of f1(f2(x, y))
            intermediate_out_offset = offset;
          } else if (BcastY) {
            intermediate_out_offset = j;
          } else {
            intermediate_out_offset = offset;
          }

          intermediate_out[intermediate_out_offset] = intermeidiate_out;
          out[offset] = compound_functor.GetOutUseIntermediateOut(
              x_val, intermeidiate_out);
        } else {
          out[offset] = compound_functor.GetOut(x_val, y_val);
        }
      }
    }
  }
}

#ifdef __NVCC__
template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActBroadcast1CUDAKernel(
    const T *x, const T *y, int h, int w, CompoundFunctor compound_functor,
    T *out, T *intermediate_out) {
  int j = blockIdx.x;
  int i = threadIdx.x;

  while (i < h) {
    int offset = i * w + j;

    T y_val = BcastY ? y[j] : y[offset];
    T x_val = BcastY ? x[offset] : x[j];
    int64_t intermediate_out_offset;

    if (KeepIntermediateOut) {
      T intermeidiate_out = compound_functor.GetIntermediateOut(x_val, y_val);

      if (SameShapeOfIntermediateOutAndOut) {
        // for the case of f1(f2(x, y))
        intermediate_out_offset = offset;
      } else if (BcastY) {
        intermediate_out_offset = j;
      } else {
        intermediate_out_offset = offset;
      }

      intermediate_out[intermediate_out_offset] = intermeidiate_out;
      out[offset] =
          compound_functor.GetOutUseIntermediateOut(x_val, intermeidiate_out);
    } else {
      out[offset] = compound_functor.GetOut(x_val, y_val);
    }

    i += ELEMWISE_MAX_BLOCK_DIM;
  }
}

template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast1CUDA(cudaStream_t stream, const T *x,
                                              const T *y,
                                              CompoundFunctor compound_functor,
                                              int h, int w, T *out,
                                              T *intermediate_out) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
  int gird_size = w;
  FusedElemwiseAndActBroadcast1CUDAKernel<
      T, CompoundFunctor, BcastY, KeepIntermediateOut,
      SameShapeOfIntermediateOutAndOut><<<gird_size, block_size, 0, stream>>>(
      x, y, h, w, compound_functor, out, intermediate_out);
}

template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActBroadcast2CUDAKernel(
    const T *x, const T *y, CompoundFunctor compound_functor, int pre, int n,
    int post, T *out, T *intermediate_out) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  while (true) {
    int i = tid / post;
    int k = tid % post;
    if (i >= pre) break;

    int offset = i * n * post + j * post + k;

    T y_val = BcastY ? y[j] : y[offset];
    T x_val = BcastY ? x[offset] : x[j];
    int64_t intermediate_out_offset;

    if (KeepIntermediateOut) {
      T intermeidiate_out = compound_functor.GetIntermediateOut(x_val, y_val);

      if (SameShapeOfIntermediateOutAndOut) {
        // for the case of f1(f2(x, y))
        intermediate_out_offset = offset;
      } else if (BcastY) {
        intermediate_out_offset = j;
      } else {
        intermediate_out_offset = offset;
      }

      intermediate_out[intermediate_out_offset] = intermeidiate_out;
      out[offset] =
          compound_functor.GetOutUseIntermediateOut(x_val, intermeidiate_out);
    } else {
      out[offset] = compound_functor.GetOut(x_val, y_val);
    }

    tid += ELEMWISE_MAX_BLOCK_DIM;
  }
}

template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast2CUDA(cudaStream_t stream, const T *x,
                                              const T *y, int pre, int n,
                                              int post,
                                              CompoundFunctor compound_functor,
                                              T *out, T *intermediate_out) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;

  FusedElemwiseAndActBroadcast2CUDAKernel<
      T, CompoundFunctor, BcastY, KeepIntermediateOut,
      SameShapeOfIntermediateOutAndOut><<<gird_size, block_size, 0, stream>>>(
      x, y, compound_functor, pre, n, post, out, intermediate_out);
}

#endif

template <typename DeviceContext, typename T, typename CompoundFunctor,
          bool KeepIntermediateOut>
void FusedElemwiseAndActComputeNoBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::Tensor &x, const framework::Tensor &y,
    CompoundFunctor compound_functor, framework::Tensor *out,
    framework::Tensor *intermediate_out) {
  size_t N = static_cast<size_t>(framework::product(x_dim));

  platform::ForRange<DeviceContext> for_range(
      ctx.template device_context<DeviceContext>(), N);

  for_range(
      FusedElemwiseAndActNoBroadcast<T, CompoundFunctor, KeepIntermediateOut>{
          x.data<T>(), y.data<T>(), compound_functor,
          out->mutable_data<T>(ctx.GetPlace()),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->mutable_data<T>(ctx.GetPlace())});
}

template <typename DeviceContext, typename T, typename CompoundFunctor,
          bool BcastY, bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActComputeWithBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim_untrimed, const framework::Tensor &x,
    const framework::Tensor &y, CompoundFunctor compound_functor, int axis,
    framework::Tensor *out, framework::Tensor *intermediate_out) {
  axis = (axis == -1 ? x_dim.size() - y_dim_untrimed.size() : axis);
  auto y_dim = trim_trailing_singular_dims(y_dim_untrimed);
  axis = (y_dim.size() == 0) ? x_dim.size() : axis;

  int pre, n, post;
  get_mid_dims(x_dim, y_dim, axis, &pre, &n, &post);

  if (post == 1) {
    int h = pre;
    int w = n;
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
      FusedElemwiseAndActBroadcast1CUDA<T, CompoundFunctor, BcastY,
                                        KeepIntermediateOut,
                                        SameShapeOfIntermediateOutAndOut>(
          ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
          y.data<T>(), compound_functor, h, w,
          out->mutable_data<T>(ctx.GetPlace()),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      FusedElemwiseAndActBroadcast1CPU<T, CompoundFunctor, BcastY,
                                       KeepIntermediateOut,
                                       SameShapeOfIntermediateOutAndOut>(
          x.data<T>(), y.data<T>(), compound_functor, h, w,
          out->mutable_data<T>(ctx.GetPlace()),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->mutable_data<T>(ctx.GetPlace()));
    }
  } else {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
      FusedElemwiseAndActBroadcast2CUDA<T, CompoundFunctor, BcastY,
                                        KeepIntermediateOut,
                                        SameShapeOfIntermediateOutAndOut>(
          ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
          y.data<T>(), pre, n, post, compound_functor,
          out->mutable_data<T>(ctx.GetPlace()),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      FusedElemwiseAndActBroadcast2CPU<T, CompoundFunctor, BcastY,
                                       KeepIntermediateOut,
                                       SameShapeOfIntermediateOutAndOut>(
          x.data<T>(), y.data<T>(), pre, n, post, compound_functor,
          out->mutable_data<T>(ctx.GetPlace()),
          intermediate_out == nullptr
              ? nullptr
              : intermediate_out->mutable_data<T>(ctx.GetPlace()));
    }
  }
}

// --- backward
template <typename T, typename DX_OP, typename DY_OP, bool UseIntermediateOut>
struct FusedElemwiseAndActGradNoBroadcast {
  HOSTDEVICE void operator()(size_t i) {
    if (dx_ != nullptr) {
      dx_[i] = UseIntermediateOut ? dx_op_(x_[i], y_[i], intermediate_out_[i],
                                           out_[i], dout_[i])
                                  : dx_op_(x_[i], y_[i], out_[i], dout_[i]);
    }
    if (dy_ != nullptr) {
      dy_[i] = UseIntermediateOut ? dy_op_(x_[i], y_[i], intermediate_out_[i],
                                           out_[i], dout_[i])
                                  : dy_op_(x_[i], y_[i], out_[i], dout_[i]);
    }
  }

  const T *x_;
  const T *y_;
  const T *intermediate_out_;
  const T *out_;
  const T *dout_;
  DX_OP dx_op_;
  DY_OP dy_op_;
  T *dx_;
  T *dy_;
};

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          bool UseIntermediateOut>
void FusedElemwiseAndActGradComputeNoBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim, const framework::Tensor *x,
    const framework::Tensor *y, const framework::Tensor *intermediate_out,
    const framework::Tensor *out, const framework::Tensor *dout, int axis,
    framework::Tensor *dx, framework::Tensor *dy, DX_OP dx_op, DY_OP dy_op) {
  size_t N = static_cast<size_t>(framework::product(x_dim));
  platform::ForRange<DeviceContext> for_range(
      ctx.template device_context<DeviceContext>(), N);
  for_range(
      FusedElemwiseAndActGradNoBroadcast<T, DX_OP, DY_OP, UseIntermediateOut>{
          x->data<T>(), y->data<T>(),
          intermediate_out ? intermediate_out->data<T>() : nullptr,
          out->data<T>(), dout->data<T>(), dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace())});
}

template <typename T, typename DX_OP, typename DY_OP, bool UseIntermediateOut,
          bool BcastY, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast1CPU(const T *x, const T *y,
                                                 const T *intermediate_out,
                                                 const T *out, const T *dout,
                                                 int h, int w, DX_OP dx_op,
                                                 DY_OP dy_op, T *dx, T *dy) {
  int64_t tmp_out_idx, x_idx, y_idx;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int offset = i * w + j;

      tmp_out_idx = BcastY ? j : offset;
      y_idx = BcastY ? j : offset;
      x_idx = BcastY ? offset : j;

      if (SameShapeOfIntermediateOutAndOut) {
        tmp_out_idx = offset;
      }

      if (dx != nullptr) {
        T tmp = UseIntermediateOut
                    ? dx_op(x[x_idx], y[y_idx], intermediate_out[tmp_out_idx],
                            out[offset], dout[offset])
                    : dx_op(x[x_idx], y[y_idx], out[offset], dout[offset]);

        if (BcastY) {
          dx[x_idx] = tmp;
        } else {
          if (i == 0) {
            dx[x_idx] = tmp;
          } else {
            dx[x_idx] += tmp;
          }
        }
      }
      if (dy != nullptr) {
        T tmp = UseIntermediateOut
                    ? dy_op(x[x_idx], y[y_idx], intermediate_out[tmp_out_idx],
                            out[offset], dout[offset])
                    : dy_op(x[x_idx], y[y_idx], out[offset], dout[offset]);
        if (BcastY) {
          if (i == 0) {
            dy[y_idx] = tmp;
          } else {
            dy[y_idx] += tmp;
          }
        } else {
          dy[y_idx] = tmp;
        }
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, bool UseIntermediateOut,
          bool BcastY, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast2CPU(const T *x, const T *y,
                                                 const T *intermediate_out,
                                                 const T *out, const T *dout,
                                                 int pre, int n, int post,
                                                 DX_OP dx_op, DY_OP dy_op,
                                                 T *dx, T *dy) {
  int64_t tmp_out_idx, x_idx, y_idx;
  for (int i = 0; i < pre; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < post; ++k) {
        int offset = i * n * post + j * post + k;

        tmp_out_idx = BcastY ? j : offset;
        y_idx = BcastY ? j : offset;
        x_idx = BcastY ? offset : j;

        if (SameShapeOfIntermediateOutAndOut) {
          tmp_out_idx = offset;
        }

        if (dx != nullptr) {
          T tmp = UseIntermediateOut
                      ? dx_op(x[x_idx], y[y_idx], intermediate_out[tmp_out_idx],
                              out[offset], dout[offset])
                      : dx_op(x[x_idx], y[y_idx], out[offset], dout[offset]);

          if (BcastY) {
            dx[x_idx] = tmp;
          } else {
            if (i == 0 && k == 0) {
              dx[x_idx] = tmp;
            } else {
              dx[x_idx] += tmp;
            }
          }
        }
        if (dy != nullptr) {
          T tmp = UseIntermediateOut
                      ? dy_op(x[x_idx], y[y_idx], intermediate_out[tmp_out_idx],
                              out[offset], dout[offset])
                      : dy_op(x[x_idx], y[y_idx], out[offset], dout[offset]);
          if (BcastY) {
            if (i == 0 && k == 0) {
              dy[y_idx] = tmp;
            } else {
              dy[y_idx] += tmp;
            }
          } else {
            dy[y_idx] = tmp;
          }
        }
      }
    }
  }
}

#ifdef __NVCC__
template <typename T, typename DX_OP, typename DY_OP, bool UseIntermediateOut,
          bool BcastY, bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActGradBroadcast1CUDAKernel(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int h, int w, DX_OP dx_op, DY_OP dy_op, T *dx, T *dy) {
  int j = blockIdx.x;
  int i = threadIdx.x;
  int tid = threadIdx.x;
  T val(0);
  int64_t tmp_out_idx, x_idx, y_idx;

  do {
    int offset = i * w + j;

    tmp_out_idx = BcastY ? j : offset;
    y_idx = BcastY ? j : offset;
    x_idx = BcastY ? offset : j;

    if (SameShapeOfIntermediateOutAndOut) {
      tmp_out_idx = offset;
    }

    if (dx != nullptr) {
      T tmp = UseIntermediateOut
                  ? dx_op(x[x_idx], y[y_idx], intermediate_out[tmp_out_idx],
                          out[offset], dout[offset])
                  : dx_op(x[x_idx], y[y_idx], out[offset], dout[offset]);

      if (BcastY) {
        dx[x_idx] = tmp;
      } else {
        val += tmp;
      }
    }
    if (dy != nullptr) {
      T tmp = UseIntermediateOut
                  ? dy_op(x[x_idx], y[y_idx], intermediate_out[tmp_out_idx],
                          out[offset], dout[offset])
                  : dy_op(x[x_idx], y[y_idx], out[offset], dout[offset]);
      if (BcastY) {
        val += tmp;
      } else {
        dy[y_idx] = tmp;
      }
    }

    i += ELEMWISE_MAX_BLOCK_DIM;
  } while (i < h);

  if (BcastY) {
    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {
    if (dx) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, bool UseIntermediateOut,
          bool BcastY, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast1CUDA(cudaStream_t stream,
                                                  const T *x, const T *y,
                                                  const T *intermediate_out,
                                                  const T *out, const T *dout,
                                                  int h, int w, DX_OP dx_op,
                                                  DY_OP dy_op, T *dx, T *dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
  int gird_size = w;
  FusedElemwiseAndActGradBroadcast1CUDAKernel<
      T, DX_OP, DY_OP, UseIntermediateOut, BcastY,
      SameShapeOfIntermediateOutAndOut><<<gird_size, block_size, 0, stream>>>(
      x, y, intermediate_out, out, dout, h, w, dx_op, dy_op, dx, dy);
}

template <typename T, typename DX_OP, typename DY_OP, bool UseIntermediateOut,
          bool BcastY, bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActGradBroadcast2CUDAKernel(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int pre, int n, int post, DX_OP dx_op, DY_OP dy_op, T *dx,
    T *dy) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  T val(0);
  int ttid = tid;
  int64_t tmp_out_idx, x_idx, y_idx;
  while (true) {
    int i = ttid / post;
    int k = ttid % post;
    if (i >= pre) break;

    int offset = i * n * post + j * post + k;

    tmp_out_idx = BcastY ? j : offset;
    y_idx = BcastY ? j : offset;
    x_idx = BcastY ? offset : j;

    if (SameShapeOfIntermediateOutAndOut) {
      tmp_out_idx = offset;
    }

    if (dx != nullptr) {
      T tmp = UseIntermediateOut
                  ? dx_op(x[x_idx], y[y_idx], intermediate_out[tmp_out_idx],
                          out[offset], dout[offset])
                  : dx_op(x[x_idx], y[y_idx], out[offset], dout[offset]);

      if (BcastY) {
        dx[x_idx] = tmp;
      } else {
        val += tmp;
      }
    }
    if (dy != nullptr) {
      T tmp = UseIntermediateOut
                  ? dy_op(x[x_idx], y[y_idx], intermediate_out[tmp_out_idx],
                          out[offset], dout[offset])
                  : dy_op(x[x_idx], y[y_idx], out[offset], dout[offset]);
      if (BcastY) {
        val += tmp;
      } else {
        dy[y_idx] = tmp;
      }
    }

    ttid += ELEMWISE_MAX_BLOCK_DIM;
  }

  if (BcastY) {
    if (dy) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {
    if (dx) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, bool UseIntermediateOut,
          bool BcastY, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast2CUDA(
    cudaStream_t stream, const T *x, const T *y, const T *intermediate_out,
    const T *out, const T *dout, int pre, int n, int post, DX_OP dx_op,
    DY_OP dy_op, T *dx, T *dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;
  FusedElemwiseAndActGradBroadcast2CUDAKernel<
      T, DX_OP, DY_OP, UseIntermediateOut, BcastY,
      SameShapeOfIntermediateOutAndOut><<<gird_size, block_size, 0, stream>>>(
      x, y, intermediate_out, out, dout, pre, n, post, dx_op, dy_op, dx, dy);
}
#endif

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActGradComputeWithBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim_untrimed, const framework::Tensor *x,
    const framework::Tensor *y, const framework::Tensor *intermediate_out,
    const framework::Tensor *out, const framework::Tensor *dout, int axis,
    framework::Tensor *dx, framework::Tensor *dy, DX_OP dx_op, DY_OP dy_op) {
  axis = (axis == -1 ? x_dim.size() - y_dim_untrimed.size() : axis);
  auto y_dim = trim_trailing_singular_dims(y_dim_untrimed);
  axis = (y_dim.size() == 0) ? x_dim.size() : axis;

  int pre, n, post;
  get_mid_dims(x_dim, y_dim, axis, &pre, &n, &post);
  if (post == 1) {
    int h = pre;
    int w = n;
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
      FusedElemwiseAndActGradBroadcast1CUDA<T, DX_OP, DY_OP, UseIntermediateOut,
                                            BcastY,
                                            SameShapeOfIntermediateOutAndOut>(
          ctx.template device_context<DeviceContext>().stream(), x->data<T>(),
          y->data<T>(),
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), h, w, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      FusedElemwiseAndActGradBroadcast1CPU<T, DX_OP, DY_OP, UseIntermediateOut,
                                           BcastY,
                                           SameShapeOfIntermediateOutAndOut>(
          x->data<T>(), y->data<T>(),
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), h, w, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
    }
  } else {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
      FusedElemwiseAndActGradBroadcast2CUDA<T, DX_OP, DY_OP, UseIntermediateOut,
                                            BcastY,
                                            SameShapeOfIntermediateOutAndOut>(
          ctx.template device_context<DeviceContext>().stream(), x->data<T>(),
          y->data<T>(),
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), pre, n, post, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      FusedElemwiseAndActGradBroadcast2CPU<T, DX_OP, DY_OP, UseIntermediateOut,
                                           BcastY,
                                           SameShapeOfIntermediateOutAndOut>(
          x->data<T>(), y->data<T>(),
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), pre, n, post, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
    }
  }
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          bool UseIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActGradComputeEx(
    const framework::ExecutionContext &ctx, const framework::Tensor *x,
    const framework::Tensor *y, const framework::Tensor *out,
    const framework::Tensor *intermediate_out, const framework::Tensor *dout,
    int axis, framework::Tensor *dx, framework::Tensor *dy, DX_OP dx_op,
    DY_OP dy_op) {
  const framework::DDim &x_dim = x->dims();
  const framework::DDim &y_dim = y->dims();
  if (UseIntermediateOut) {
    PADDLE_ENFORCE(intermediate_out, "intermediate_out should not be nullptr");
  }
  if (x_dim == y_dim) {
    FusedElemwiseAndActGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP,
                                              UseIntermediateOut>(
        ctx, x_dim, y_dim, x, y, intermediate_out, out, dout, axis, dx, dy,
        dx_op, dy_op);
  } else {  // Y is a scalar
    bool bcast_y = x_dim.size() >= y_dim.size();
    if (x_dim.size() == y_dim.size()) {
      for (int i = 0; i < x_dim.size(); ++i) {
        if (x_dim[i] < y_dim[i]) {
          bcast_y = false;
          break;
        }
      }
    }

    // z = f1(x, f2(y))
    // z = f1(f2(x, y))
    if (bcast_y) {  // Y should be broadcast.
      FusedElemwiseAndActGradComputeWithBroadcast<
          DeviceContext, T, DX_OP, DY_OP, UseIntermediateOut, true /*BcastY*/,
          SameShapeOfIntermediateOutAndOut>(ctx, x_dim, y_dim, x, y,
                                            intermediate_out, out, dout, axis,
                                            dx, dy, dx_op, dy_op);
    } else {
      FusedElemwiseAndActGradComputeWithBroadcast<
          DeviceContext, T, DX_OP, DY_OP, UseIntermediateOut, false /*BcastY*/,
          SameShapeOfIntermediateOutAndOut>(ctx, y_dim, x_dim, x, y,
                                            intermediate_out, out, dout, axis,
                                            dx, dy, dx_op, dy_op);
    }
  }
}

template <typename DeviceContext, typename T, typename CompoundFunctor,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActComputeEx(const framework::ExecutionContext &ctx,
                                  const framework::Tensor &x,
                                  const framework::Tensor &y, int axis,
                                  CompoundFunctor compound_functor,
                                  framework::Tensor *out,
                                  framework::Tensor *intermediate_out) {
  if (KeepIntermediateOut) {
    PADDLE_ENFORCE(intermediate_out,
                   "The keep_intermediate_value is opened, "
                   "intermediate_out should not be nullptr.");
  }

  const framework::DDim &x_dim = x.dims();
  const framework::DDim &y_dim = y.dims();
  if (x.dims() == y.dims()) {
    FusedElemwiseAndActComputeNoBroadcast<DeviceContext, T, CompoundFunctor,
                                          KeepIntermediateOut>(
        ctx, x_dim, x, y, compound_functor, out, intermediate_out);
  } else {
    // Whether the shape of Y is a continuous subsequence of X,
    // For more information please refer to the op's introduction.
    bool bcast_y = x.dims().size() >= y.dims().size();
    if (x.dims().size() == y.dims().size()) {
      for (int i = 0; i < x.dims().size(); ++i) {
        if (x.dims()[i] < y.dims()[i]) {
          bcast_y = false;
          break;
        }
      }
    }

    // z = f1(x, f2(y))
    // z = f1(f2(x, y))
    if (bcast_y) {  // Y should be broadcast.
      // In this case,
      // for 'f2(y)', the shape of intermediate_out should be equal to the shape
      // of Y.
      // for 'f2(x, y)', the shape of intermediate_out should be equal to the
      // shape of Out.
      // the shape of Out should be equal to the shape of X.
      FusedElemwiseAndActComputeWithBroadcast<
          DeviceContext, T, CompoundFunctor, true /*BcastY*/,
          KeepIntermediateOut, SameShapeOfIntermediateOutAndOut>(
          ctx, x_dim /*OutShape*/, y_dim, x, y, compound_functor, axis, out,
          intermediate_out);
    } else {
      // In this case,
      // for 'f2(y)', the shape of intermediate_out should be equal to the shape
      // of Out.
      // for 'f2(x, y)', the shape of intermediate_out should be equal to the
      // shape of Out.
      // the shape of Out should be equal to the shape of Y.
      FusedElemwiseAndActComputeWithBroadcast<
          DeviceContext, T, CompoundFunctor, false /*BcastY*/,
          KeepIntermediateOut, SameShapeOfIntermediateOutAndOut>(
          ctx, y_dim /*OutShape*/, x_dim, x, y, compound_functor, axis, out,
          intermediate_out);
    }
  }
}
}  // namespace operators
}  // namespace paddle
