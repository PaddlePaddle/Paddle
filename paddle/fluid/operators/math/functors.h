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

#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

struct Functor {};

struct IdentityFunctor : public Functor {
  // y = x;
  template <typename T>
  inline HOSTDEVICE T operator()(T ele) const {
    return ele;
  }
};

// AddFunctor
struct AddFunctor : public Functor {
  explicit AddFunctor(Functor* fun) : func_tail1_(fun) {}
  explicit AddFunctor(Functor* fun1, Functor* fun2)
      : func_tail1_(fun1), func_tail2_(fun2) {}

  // out = f(x + y);
  template <typename T>
  inline HOSTDEVICE T operator()(T a, T b) const {
    return (*func_tail1_)(a + b);
  }

  // out = f2(f1(x, y) + z);
  template <typename T>
  inline HOSTDEVICE T operator()(T a, T b, T c) const {
    return (*func_tail2_)((*func_tail1_)(a + b), c);
  }

 private:
  std::unique_ptr<Functor> func_tail1_;
  std::unique_ptr<Functor> func_tail2_;
};

struct AddGradFunctor : public Functor {
  explicit AddGradFunctor(Functor* fun) : func_tail1_(fun) {}
  explicit AddGradFunctor(Functor* fun1, Functor* fun2)
      : func_tail1_(fun1), func_tail2_(fun2) {}

  // dx = dout * f'(x, y); dy = dout * f'(x, y);
  template <typename T>
  inline HOSTDEVICE T operator()(T a, T b, T out, T dout) const {
    return (*func_tail1_)(a, b, out, dout);
  }

  // dx = dout * f2'(f1(x, y), z) * f1'(x, y);
  // dy = dout * f2'(f1(x, y), z) * f1'(x, y);
  // dz = dout * f2'(f1(x, y), z);
  template <typename T>
  inline HOSTDEVICE T operator()(T a, T b, T c, T out, T dout) const {
    // TODO(zcd): analysis the backward.
    return (*func_tail1_)(dout);
  }

  std::unique_ptr<Functor> func_tail1_;
  std::unique_ptr<Functor> func_tail2_;
};

// ScaleFunctor
struct ScaleFunctor : public Functor {
  explicit ScaleFunctor(int64_t coeff, Functor* fun)
      : coeff_(coeff), func_tail_(fun) {}

  // out = scale * x;
  template <typename T>
  inline HOSTDEVICE T operator()(T ele) const {
    return (*func_tail_)(ele * static_cast<T>(coeff_));
  }

 private:
  int64_t coeff_;
  std::unique_ptr<Functor> func_tail_;
};

struct ScaleGradFunctor : public Functor {
  explicit ScaleGradFunctor(int64_t coeff, Functor* fun)
      : coeff_(coeff), func_tail_(fun) {}

  // out = dout * scale;
  template <typename T>
  inline HOSTDEVICE T operator()(T a) const {
    return (*func_tail_)(a)*coeff_;
  }

 private:
  int64_t coeff_;
  std::unique_ptr<Functor> func_tail_;
};

// ReluFunctor
struct ReluFunctor : public Functor {
  explicit ReluFunctor(int64_t coeff, Functor* fun)
      : coeff_(coeff), func_tail_(fun) {}

  // out = scale * x;
  template <typename T>
  inline HOSTDEVICE T operator()(T ele) const {
    return (*func_tail_)(ele * static_cast<T>(coeff_));
  }

 private:
  int64_t coeff_;
  std::unique_ptr<Functor> func_tail_;
};

struct ReluGradFunctor : public Functor {
  explicit ReluGradFunctor(int64_t coeff, Functor* fun)
      : coeff_(coeff), func_tail_(fun) {}

  // out = dout * scale;
  template <typename T>
  inline HOSTDEVICE T operator()(T a) const {
    return (*func_tail_)(a)*coeff_;
  }

 private:
  int64_t coeff_;
  std::unique_ptr<Functor> func_tail_;
};

//------------
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
inline void get_mid_dims(const framework::DDim& x_dims,
                         const framework::DDim& y_dims, const int axis,
                         int* pre, int* n, int* post) {
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

inline void trim_trailing_singular_dims(framework::DDim* dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims->size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if ((*dims)[actual_dims_size - 1] != 1) break;
  }
  if (actual_dims_size != dims->size()) {
    auto actual_dims = framework::vectorize(*dims);
    actual_dims.resize(actual_dims_size);
    *dims = framework::make_ddim(actual_dims);
  }
}

template <typename T, typename DeviceContext>
class EleRowwiseTransformIterator;
template <typename T, typename DeviceContext>
class EleMidWiseTransformIterator;

template <typename T>
class EleRowwiseTransformIterator<T, platform::CPUDeviceContext> {
 public:
  EleRowwiseTransformIterator(const T* ptr, int n) : ptr_(ptr), i_(0), n_(n) {}

  EleRowwiseTransformIterator<T, platform::CPUDeviceContext>& operator++() {
    ++i_;
    if (UNLIKELY(i_ == n_)) {
      i_ = 0;
    }
    return *this;
  }

  bool operator==(
      const EleRowwiseTransformIterator<T, platform::CPUDeviceContext>& rhs)
      const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(
      const EleRowwiseTransformIterator<T, platform::CPUDeviceContext>& rhs)
      const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T& operator*() { return ptr_[i_]; }

 private:
  const T* ptr_;
  int i_;
  int64_t n_;
};

template <typename T>
class EleMidWiseTransformIterator<T, platform::CPUDeviceContext> {
 public:
  EleMidWiseTransformIterator(const T* ptr, int n, int post)
      : ptr_(ptr), i_(0), j_(0), n_(n), post_(post) {}

  EleMidWiseTransformIterator<T, platform::CPUDeviceContext>& operator++() {
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

  bool operator==(
      const EleMidWiseTransformIterator<T, platform::CPUDeviceContext>& rhs)
      const {
    return (ptr_ + i_) == &(*rhs);
  }

  bool operator!=(
      const EleMidWiseTransformIterator<T, platform::CPUDeviceContext>& rhs)
      const {
    return (ptr_ + i_) != &(*rhs);
  }

  const T& operator*() { return ptr_[i_]; }

 private:
  const T* ptr_;
  int64_t i_;
  int64_t j_;
  int64_t n_;
  int64_t post_;
};

#ifdef __NVCC__
template <typename T>
class EleRowwiseTransformIterator<T, platform::CUDADeviceContext>
    : public thrust::iterator_adaptor<
          EleRowwiseTransformIterator<T, platform::CUDADeviceContext>,
          const T*> {
 public:
  typedef thrust::iterator_adaptor<
      EleRowwiseTransformIterator<T, platform::CUDADeviceContext>, const T*>
      super_t;
  HOSTDEVICE EleRowwiseTransformIterator(const T* x, int n)
      : super_t(x), begin_(x), n_(n) {}
  friend class thrust::iterator_core_access;

 private:
  unsigned int n_;
  const T* begin_;
  HOSTDEVICE typename super_t::reference dereference() const {
    return *(begin_ + (this->base() - begin_) % n_);
  }
};

template <typename T>
class EleMidWiseTransformIterator<T, platform::CUDADeviceContext>
    : public thrust::iterator_adaptor<
          EleMidWiseTransformIterator<T, platform::CUDADeviceContext>,
          const T*> {
 public:
  typedef thrust::iterator_adaptor<
      EleMidWiseTransformIterator<T, platform::CUDADeviceContext>, const T*>
      super_t;
  HOSTDEVICE EleMidWiseTransformIterator(const T* x, int n, int post)
      : super_t(x), begin_(x), n_(n), post_(post) {}
  friend class thrust::iterator_core_access;

 private:
  unsigned int post_;
  unsigned int n_;
  const T* begin_;
  HOSTDEVICE typename super_t::reference dereference() const {
    return *(begin_ + (((this->base() - begin_) / post_) % n_));
  }
};
#endif

template <typename Functor, typename T, typename DeviceContext,
          typename OutType = T>
class TransformFunctor {
 public:
  TransformFunctor(const framework::Tensor* x, const framework::Tensor* y,
                   framework::Tensor* z, const DeviceContext& ctx, Functor func)
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
    trans(ctx_, x_, x_ + nx_,
          EleRowwiseTransformIterator<T, DeviceContext>(y_, n), z_, func_);
  }

  inline void RunMidWise(int n, int pre, int post) const {
    platform::Transform<DeviceContext> trans;
    trans(ctx_, x_, x_ + nx_,
          EleMidWiseTransformIterator<T, DeviceContext>(y_, n, post), z_,
          func_);
  }

 private:
  const T* x_;
  const T* y_;
  OutType* z_;
  int64_t nx_;
  const DeviceContext& ctx_;
  Functor func_;
};

template <typename T, typename DX_OP, typename DY_OP>
struct ElemwiseGradNoBroadcast {
  const T* x_;
  const T* y_;
  const T* out_;
  const T* dout_;

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
  T* dx_;
  T* dy_;
};

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast1CPU(const T* x, const T* y, const T* out,
                                      const T* dout, int h, int w, DX_OP dx_op,
                                      DY_OP dy_op, T* dx, T* dy) {
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
    const T* x, const T* y, const T* out, const T* dout, int h, int w,
    DX_OP dx_op, DY_OP dy_op, T* dx, T* dy) {
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
static void ElemwiseGradBroadcast1CUDA(cudaStream_t stream, const T* x,
                                       const T* y, const T* out, const T* dout,
                                       int h, int w, DX_OP dx_op, DY_OP dy_op,
                                       T* dx, T* dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
  int gird_size = w;
  ElemwiseGradBroadcast1CUDAKernel<<<gird_size, block_size, 0, stream>>>(
      x, y, out, dout, h, w, dx_op, dy_op, dx, dy);
}

#endif

template <typename T, typename DX_OP, typename DY_OP>
static void ElemwiseGradBroadcast2CPU(const T* x, const T* y, const T* out,
                                      const T* dout, int pre, int n, int post,
                                      DX_OP dx_op, DY_OP dy_op, T* dx, T* dy) {
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
    const T* x, const T* y, const T* out, const T* dout, int pre, int n,
    int post, DX_OP dx_op, DY_OP dy_op, T* dx, T* dy) {
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
static void ElemwiseGradBroadcast2CUDA(cudaStream_t stream, const T* x,
                                       const T* y, const T* out, const T* dout,
                                       int pre, int n, int post, DX_OP dx_op,
                                       DY_OP dy_op, T* dx, T* dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;
  ElemwiseGradBroadcast2CUDAKernel<<<gird_size, block_size, 0, stream>>>(
      x, y, out, dout, pre, n, post, dx_op, dy_op, dx, dy);
}

#endif

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseGradCompute(const framework::ExecutionContext& ctx,
                         const framework::Tensor& x, const framework::Tensor& y,
                         const framework::Tensor& out,
                         const framework::Tensor& dout, int axis,
                         framework::Tensor* dx, framework::Tensor* dy,
                         DX_OP dx_op, DY_OP dy_op) {
  if (x.dims() == y.dims()) {
    size_t N = static_cast<size_t>(framework::product(x.dims()));
    platform::ForRange<DeviceContext> for_range(
        ctx.template device_context<DeviceContext>(), N);
    for_range(ElemwiseGradNoBroadcast<T, DX_OP, DY_OP>{
        x.data<T>(), y.data<T>(), out.data<T>(), dout.data<T>(), dx_op, dy_op,
        dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
        dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace())});
  } else {  // Y is a scalar
    auto x_dim = x.dims();
    auto y_dim = y.dims();

    axis = (axis == -1 ? x_dim.size() - y_dim.size() : axis);
    trim_trailing_singular_dims(&y_dim);
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
            x.data<T>(), y.data<T>(), out.data<T>(), dout.data<T>(), h, w,
            dx_op, dy_op,
            dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
            dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
      }
    } else {
      if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef __NVCC__
        ElemwiseGradBroadcast2CUDA(
            ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
            y.data<T>(), out.data<T>(), dout.data<T>(), pre, n, post, dx_op,
            dy_op,
            dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
            dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
#endif
      } else {
        ElemwiseGradBroadcast2CPU(
            x.data<T>(), y.data<T>(), out.data<T>(), dout.data<T>(), pre, n,
            post, dx_op, dy_op,
            dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
            dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
      }
    }
  }
}

template <typename Functor, typename DeviceContext, typename T,
          typename OutType = T>
void ElementwiseComputeEx(const framework::ExecutionContext& ctx,
                          const framework::Tensor* x,
                          const framework::Tensor* y, int axis, Functor func,
                          framework::Tensor* z) {
  TransformFunctor<Functor, T, DeviceContext, OutType> functor(
      x, y, z, ctx.template device_context<DeviceContext>(), func);

  auto x_dims = x->dims();
  auto y_dims = y->dims();
  PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                    "Rank of first input must >= rank of second input.");

  if (x_dims == y_dims) {
    functor.Run();
    return;
  }

  axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
  PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                 "Axis should be in range [0, x_dims)");
  trim_trailing_singular_dims(&y_dims);
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

}  // namespace math
}  // namespace operators
}  // namespace paddle
