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

#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/empty_kernel.h"
#include "paddle/pten/kernels/funcs/math_function.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/function_traits.h"
#include "paddle/pten/backends/gpu/gpu_launch_config.h"
#include "paddle/pten/kernels/primitive/kernel_primitives.h"

namespace kps = pten::kps;

#endif

namespace pten {

enum ElementwiseType { kUnary = 1, kBinary = 2, kTernary = 3, kAny = -1 };
/* Packing scalar type T(float, int etc.) into Array<T, NumOuts> type
   for supporting multiple-output feature in elementwise system.*/
template <class T, int Num>
using ConditionalT =
    typename std::conditional_t<Num == 1, T, pten::framework::Array<T, Num>>;

namespace funcs {
using DDim = pten::framework::DDim;

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
struct ElemwiseGradNoBroadcast {
  const T *x_;
  const T *y_;
  const Tout *out_;
  const Tout *dout_;

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
        z_(ctx.template Alloc<OutType>(z)),
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
    return DDim(pten::framework::make_dim());
  }
  DDim actual_dims = pten::framework::make_ddim(trim_dims);
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

// for broadcast backwards
static inline std::vector<int> GetReduceDim(const paddle::framework::DDim &in,
                                            const paddle::framework::DDim &out,
                                            int axis) {
  axis =
      (axis == -1 ? std::abs(static_cast<int>(out.size() - in.size())) : axis);
  std::vector<int> dims;
  for (int i = 0; i < axis; ++i) {
    dims.push_back(i);
  }
  for (int i = 0; i < in.size(); ++i) {
    if (out[i + axis] != in[i]) {
      dims.push_back(i + axis);
    }
  }
  for (int i = axis + in.size(); i < out.size(); ++i) {
    dims.push_back(i);
  }
  return dims;
}

template <typename DeviceContext, typename T>
static inline void GetDoubleGradSafeTensor(const DeviceContext &dev_ctx,
                                           const DenseTensor &x,
                                           const DenseTensor *ddx,
                                           DenseTensor *ddx_safe) {
  if (ddx) {
    *ddx_safe = *ddx;
  } else {
    auto meta = pten::DenseTensorMeta(x.dtype(), x.dims(), x.layout());
    *ddx_safe = pten::Empty(dev_ctx, std::move(meta));
    ddx_safe->mutable_data(dev_ctx.GetPlace());
    pten::funcs::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, ddx_safe, static_cast<T>(0));
  }
}

template <typename DeviceContext,
          typename T,
          typename DX_OP,
          typename DY_OP,
          typename Tout = T>
void ElemwiseGradComputeNoBroadcast(const DeviceContext &dev_ctx,
                                    const DDim &x_dim,
                                    const DDim &y_dim,
                                    const DenseTensor &x,
                                    const DenseTensor &y,
                                    const DenseTensor &out,
                                    const DenseTensor &dout,
                                    int axis,
                                    DenseTensor *dx,
                                    DenseTensor *dy,
                                    DX_OP dx_op,
                                    DY_OP dy_op) {
  size_t N = static_cast<size_t>(pten::framework::product(x_dim));
  paddle::platform::ForRange<DeviceContext> for_range(dev_ctx, N);
  for_range(ElemwiseGradNoBroadcast<T, DX_OP, DY_OP, Tout>{
      x.data<T>(),
      y.data<T>(),
      out.data<Tout>(),
      dout.data<Tout>(),
      dx_op,
      dy_op,
      dx == nullptr ? nullptr : dev_ctx.template Alloc<T>(dx),
      dy == nullptr ? nullptr : dev_ctx.template Alloc<T>(dy)});
}

inline void ElementwiseGradPreProcess(const DenseTensor &dout,
                                      DenseTensor *dx) {
  if (dx != nullptr) {
    dx->set_lod(dout.lod());
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)

// static unroller
template <template <int Index, int VecSize> typename Func,
          int VecSize,
          int End,
          int Begin = 0>
struct Unroller {
  template <typename... Args>
  static HOSTDEVICE inline void step(Args &&... args) {
    Func<Begin, VecSize>::Apply(std::forward<Args>(args)...);
    Unroller<Func, VecSize, End, Begin + 1>::step(args...);
  }
};

template <template <int Index, int VecSize> typename Func, int VecSize, int End>
struct Unroller<Func, VecSize, End, End> {
  template <typename... Args>
  static HOSTDEVICE inline void step(Args &&... args) {}
};

template <int Index, int VecSize>
struct Loader {
  template <typename Array, typename ArgsT>
  static __device__ void Apply(const Array &in,
                               ArgsT *args,
                               int num,
                               int data_offset,
                               bool is_boundary) {
    using Type = std::tuple_element_t<Index, ArgsT>;
    kps::Init<Type, ArgsT, Index, VecSize>(args, static_cast<Type>(1.0f));
    if (is_boundary) {
      kps::ReadData<Type, VecSize, 1, 1, ArgsT, Index, true>(
          args, reinterpret_cast<const Type *>(in[Index]) + data_offset, num);
    } else {
      kps::ReadData<Type, VecSize, 1, 1, ArgsT, Index, false>(
          args, reinterpret_cast<const Type *>(in[Index]) + data_offset, num);
    }
  }
};

template <int Index, int VecSize>
struct InputSetter {
  template <typename Array>
  static HOSTDEVICE void Apply(
      const std::vector<const DenseTensor *> &ins_tensor, Array *ins_data) {
    (*ins_data)[Index] =
        reinterpret_cast<const _ptr_ char *>(ins_tensor[Index]->data());
  }
};

template <int Index, int VecSize>
struct VecSizeGetter {
  template <typename ArgsT>
  static HOSTDEVICE void Apply(const std::vector<const DenseTensor *> &ins,
                               const ArgsT &args,
                               int *vec_size) {
    using Type = std::tuple_element_t<Index, ArgsT>;
    *vec_size = std::min<int>(
        *vec_size,
        paddle::platform::GetVectorizedSize(ins[Index]->data<Type>()));
  }
};

template <typename OutT, typename Functor>
int GetVectorizedSizeForTensors(const std::vector<const DenseTensor *> &ins,
                                const std::vector<DenseTensor *> &outs) {
  using Traits = paddle::platform::FunctionTraits<Functor>;
  using ArgsT = typename Traits::ArgsTuple;
  const int Arity = Traits::arity;
  int vec_size = 4;
  ArgsT arg;
  // The Arg VecSize=1 is to match the Unroller template.
  Unroller<VecSizeGetter, 1, Arity>::step(ins, arg, &vec_size);
  for (auto iter = outs.begin(); iter != outs.end(); ++iter) {
    vec_size = std::min<int>(
        vec_size, paddle::platform::GetVectorizedSize((*iter)->data<OutT>()));
  }
  return vec_size;
}

template <typename InT,
          typename OutT,
          int VecSize,
          typename Functor,
          int Arity,
          bool CallElementwiseAny = false>
struct ElementwisePrimitiveCaller {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result);
};

template <typename InT, typename OutT, int VecSize, typename Functor, int Arity>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, Arity, true> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseAny<InT, OutT, VecSize, 1, 1, Arity, Functor>(
        result, args, func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 0, false> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseConstant<InT, OutT, VecSize, 1, 1, Functor>(result, func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 1, false> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseUnary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 2, false> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], args[1], func);
  }
};

template <typename InT, typename OutT, int VecSize, typename Functor>
struct ElementwisePrimitiveCaller<InT, OutT, VecSize, Functor, 3, false> {
  __device__ inline void operator()(Functor func,
                                    InT (*args)[VecSize],
                                    OutT *result) {
    kps::ElementwiseTernary<InT, OutT, VecSize, 1, 1, Functor>(
        result, args[0], args[1], args[2], func);
  }
};

namespace detail {
template <class F, class Tuple, std::size_t... Index>
// GCC/Clang need the decltype() return type
HOSTDEVICE constexpr decltype(auto) ApplyImpl(F &&f,
                                              Tuple &&t,
                                              std::index_sequence<Index...>) {
  return std::forward<F>(f)(std::get<Index>(std::forward<Tuple>(t))...);
}
}  // namespace detail

template <class F, class Tuple>
HOSTDEVICE constexpr decltype(auto) Apply(F &&f, Tuple &&t) {
  return detail::ApplyImpl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template <typename OutT,
          int VecSize,
          typename Functor,
          typename ArgsT,
          int Arity>
struct SameDimsElementwisePrimitiveCaller {
  __device__ inline void operator()(Functor func, ArgsT *args, OutT *result) {
#pragma unroll
    for (int idx = 0; idx < VecSize; ++idx) {
      result[idx] = static_cast<OutT>(Apply(func, args[idx]));
    }
  }
};

template <typename OutT, int VecSize, bool IsBoundary, int NumOuts>
struct ElementwiseWriteDataCaller {
  __device__ __forceinline__ void operator()(
      pten::framework::Array<_ptr_ OutT *, NumOuts> outs,
      ConditionalT<OutT, NumOuts> src[VecSize],
      int block_offset,
      int num) {
    OutT dst[NumOuts][VecSize];
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
#pragma unroll
      for (int j = 0; j < NumOuts; ++j) {
        dst[j][i] = (src[i])[j];
      }
    }
#pragma unroll
    for (int i = 0; i < NumOuts; ++i) {
      kps::WriteData<OutT, VecSize, 1, 1, IsBoundary>(
          outs[i] + block_offset, dst[i], num);
    }
  }
};

template <typename OutT, int VecSize, bool IsBoundary>
struct ElementwiseWriteDataCaller<OutT, VecSize, IsBoundary, 1> {
  __device__ __forceinline__ void operator()(
      pten::framework::Array<_ptr_ OutT *, 1> outs,
      OutT src[VecSize],
      int block_offset,
      int num) {
    kps::WriteData<OutT, VecSize, 1, 1, IsBoundary>(
        outs[0] + block_offset, src, num);
  }
};

template <typename OutT,
          typename Functor,
          int Arity,
          int NumOuts,
          int VecSize,
          bool IsBoundary>
__device__ void VectorizedElementwiseKernelImpl(

    const pten::framework::Array<const _ptr_ char *__restrict__, Arity> &in,
    pten::framework::Array<_ptr_ OutT *, NumOuts> outs,
    int num,
    int data_offset,
    Functor func) {
  using Traits = paddle::platform::FunctionTraits<Functor>;
  using ArgsT = typename Traits::ArgsTuple;
  ArgsT args[VecSize];
  ConditionalT<OutT, NumOuts> result[VecSize];

  Unroller<Loader, VecSize, Arity>::step(
      in, args, num, data_offset, IsBoundary);

  SameDimsElementwisePrimitiveCaller<ConditionalT<OutT, NumOuts>,
                                     VecSize,
                                     Functor,
                                     ArgsT,
                                     Arity>()(func, args, result);

  ElementwiseWriteDataCaller<OutT, VecSize, IsBoundary, NumOuts>()(
      outs, result, data_offset, num);
}

template <typename OutT, typename Functor, int Arity, int NumOuts, int VecSize>
__global__ void VectorizedElementwiseKernel(
    pten::framework::Array<const _ptr_ char *__restrict__, Arity> ins,
    pten::framework::Array<_ptr_ OutT *, NumOuts> outs,
    int size,
    int main_offset,
    Functor func) {
  int data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  int stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  for (; data_offset < main_offset; data_offset += stride) {
    VectorizedElementwiseKernelImpl<OutT,
                                    Functor,
                                    Arity,
                                    NumOuts,
                                    VecSize,
                                    false>(
        ins, outs, VecSize * BLOCK_NUM_X, data_offset, func);
  }

  int num = size - data_offset;
  if (num > 0) {
    VectorizedElementwiseKernelImpl<OutT,
                                    Functor,
                                    Arity,
                                    NumOuts,
                                    VecSize,
                                    true>(ins, outs, num, data_offset, func);
  }
}

template <typename OutT, typename Functor, int Arity, int NumOuts, int VecSize>
void ElementwiseCudaKernel(const KPDevice &ctx,
                           const std::vector<const DenseTensor *> &ins,
                           std::vector<DenseTensor *> *outs,
                           Functor func) {
  auto numel =
      (*outs)[0]->numel();  // To avoid running errors when ins.size()== 0
  pten::framework::Array<const _ptr_ char *__restrict__, Arity> ins_data;
  pten::framework::Array<_ptr_ OutT *, NumOuts> outs_data;

  Unroller<InputSetter, VecSize, Arity>::step(ins, &ins_data);
  for (int i = 0; i < NumOuts; ++i) {
    outs_data[i] = ctx.Alloc<OutT>((*outs)[i]);
  }
#ifdef PADDLE_WITH_XPU2
  int block_size = 64;
  int grid_size = 8;
  auto stream = ctx.x_context()->xpu_stream;
  int main_offset = (numel / (VecSize * block_size)) * VecSize * block_size;
  VectorizedElementwiseKernel<OutT,
                              Functor,
                              Arity,
                              NumOuts,
                              VecSize><<<grid_size, block_size, 0, stream>>>(
      ins_data, outs_data, numel, main_offset, func);
#else
  auto gpu_config =
      pten::backends::gpu::GetGpuLaunchConfig1D(ctx, numel, VecSize);
  int main_offset = (numel / (VecSize * gpu_config.GetBlockSize())) * VecSize *
                    gpu_config.GetBlockSize();
  auto stream = ctx.stream();
  VectorizedElementwiseKernel<OutT, Functor, Arity, NumOuts, VecSize><<<
      gpu_config.block_per_grid,
      gpu_config.thread_per_block,
      0,
      stream>>>(ins_data, outs_data, numel, main_offset, func);
#endif
}

template <typename OutT, typename Functor, int NumOuts = 1>
void LaunchSameDimsElementwiseCudaKernel(
    const KPDevice &ctx,
    const std::vector<const DenseTensor *> &ins,
    std::vector<DenseTensor *> *outs,
    Functor func) {
  using Traits = paddle::platform::FunctionTraits<Functor>;
  const int kArity = Traits::arity;
  PADDLE_ENFORCE_EQ(ins.size(),
                    kArity,
                    paddle::platform::errors::InvalidArgument(
                        "The number of inputs is expected to be equal to the "
                        "arity of functor. But recieved: the number of inputs "
                        "is %d, the arity of functor is %d.",
                        ins.size(),
                        kArity));
  PADDLE_ENFORCE_EQ(outs->size(),
                    NumOuts,
                    paddle::platform::errors::InvalidArgument(
                        "Number of outputs shall equal to number of functions, "
                        "but number of outputs is %d, of functions is %d.",
                        outs->size(),
                        NumOuts));

  if (NumOuts > 1) {
    for (int i = 1; i < NumOuts; ++i) {
      PADDLE_ENFORCE_EQ(
          (*outs)[i]->dims(),
          (*outs)[0]->dims(),
          paddle::platform::errors::InvalidArgument(
              "The shape of each output tensor shall be identical yet, "
              "but %dth output tensor`s shape is not.",
              i));
    }
  }

  // calculate the max vec_size for all ins and outs
  int vec_size = GetVectorizedSizeForTensors<OutT, Functor>(ins, *outs);
  switch (vec_size) {
    case 4:
      ElementwiseCudaKernel<OutT, Functor, kArity, NumOuts, 4>(
          ctx, ins, outs, func);
      break;
    case 2:
      ElementwiseCudaKernel<OutT, Functor, kArity, NumOuts, 2>(
          ctx, ins, outs, func);
      break;
    case 1:
      ElementwiseCudaKernel<OutT, Functor, kArity, NumOuts, 1>(
          ctx, ins, outs, func);
      break;
    default: {
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}
#endif

}  // namespace funcs
}  // namespace pten
