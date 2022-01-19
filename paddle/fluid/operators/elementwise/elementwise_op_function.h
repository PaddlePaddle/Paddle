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
#include <functional>  // for multiplies
#include <iterator>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/elementwise/elementwise_functor.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/transform.h"

#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/kernels/cpu/elementwise.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#ifdef __NVCC__
#include <cuda.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#include <thrust/iterator/iterator_adaptor.h>

#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

#endif

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

namespace paddle {
namespace operators {

/*
*  Pack input and output tensors into respective vectors with
*  consideration of varible X`s class type.
*  Input variable X is supported to be whether LoDTensor or
*  SelectedRows class type in this package function, once X
*  was SelectedRows type, a valid pointer x_for_selectedrows
*  is excepted to be passed in from op kernel for acquisition
*  of the valid address of LoDTensor created ahead in the function.
*/
template <typename OutT>
int PackTensorsIntoVector(const framework::ExecutionContext &ctx,
                          std::vector<const framework::Tensor *> *ins,
                          std::vector<framework::Tensor *> *outs,
                          framework::Tensor *x_for_selectedrows = nullptr) {
  int axis = -1;
  auto x_var = ctx.InputVar("X");
  PADDLE_ENFORCE_NOT_NULL(
      x_var, platform::errors::InvalidArgument(
                 "Unable to get input Variable X, Variable name is %s.\n",
                 ctx.InputName("X")));
  auto *y = ctx.Input<framework::LoDTensor>("Y");
  framework::Tensor *z;

  if (x_var->IsType<framework::LoDTensor>()) {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    z = ctx.Output<framework::LoDTensor>("Out");
    ins->emplace_back(x);
  } else if (x_var->IsType<framework::SelectedRows>()) {
    PADDLE_ENFORCE_EQ(y->dims().size() == 1 && y->dims()[0] == 1, true,
                      platform::errors::InvalidArgument(
                          "For elementwise_op, if X is Sparse, Y must be "
                          "scalar. But reveived the size of Y = %d.",
                          y->dims().size()));
    PADDLE_ENFORCE_NOT_NULL(
        x_for_selectedrows,
        platform::errors::InvalidArgument(
            "The parameter x_for_selectedrows is excepted to "
            "be valid, once input varible X`s class type is "
            "SelectedRows.\n"));
    auto &x_sele = x_var->Get<framework::SelectedRows>();
    auto out_sele = ctx.Output<framework::SelectedRows>("Out");
    *x_for_selectedrows = x_sele.value();
    out_sele->set_rows(x_sele.rows());
    out_sele->set_height(x_sele.height());
    out_sele->mutable_value()->Resize(x_sele.value().dims());
    out_sele->mutable_value()->mutable_data(ctx.GetPlace(),
                                            x_for_selectedrows->type());
    z = ctx.Output<framework::SelectedRows>("Out")->mutable_value();
    ins->emplace_back(x_for_selectedrows);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "X's type[%s] is not supported by elementwise_op. X's type should be "
        "LoDTensor or SelectedRows.",
        framework::ToTypeName(x_var->Type())));
  }
  z->mutable_data<OutT>(ctx.GetPlace());
  outs->emplace_back(z);

  if (y != nullptr) {
    ins->emplace_back(y);
    axis = ctx.HasAttr("axis") ? ctx.Attr<int>("axis") : -1;
  }
  return axis;
}

inline void GetBroadcastDimsArrays(const framework::DDim &x_dims,
                                   const framework::DDim &y_dims,
                                   int *x_dims_array, int *y_dims_array,
                                   int *out_dims_array, const int max_dim,
                                   const int axis) {
  pten::funcs::GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array,
                                      y_dims_array, out_dims_array, max_dim,
                                      axis);
}

inline framework::DDim trim_trailing_singular_dims(
    const framework::DDim &dims) {
  return pten::funcs::trim_trailing_singular_dims(dims);
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          typename Tout = T>
void ElemwiseGradCompute(const framework::ExecutionContext &ctx,
                         const framework::Tensor &x, const framework::Tensor &y,
                         const framework::Tensor &out,
                         const framework::Tensor &dout, int axis,
                         framework::Tensor *dx, framework::Tensor *dy,
                         DX_OP dx_op, DY_OP dy_op) {
  const framework::DDim &x_dim = x.dims();
  const framework::DDim &y_dim = y.dims();
  const auto &dev_ctx = ctx.template device_context<DeviceContext>();
  if (x.dims() == y.dims()) {
    pten::funcs::ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP,
                                                Tout>(
        dev_ctx, x_dim, y_dim, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
  } else {
    pten::ElemwiseGradComputeWithBroadcast<T, DX_OP, DY_OP, Tout>(
        dev_ctx, x_dim, y_dim, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
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
  const framework::DDim &x_dim = x.dims();
  const framework::DDim &y_dim = y.dims();
  const auto &dev_ctx = ctx.template device_context<DeviceContext>();
  if (x.dims() == y.dims()) {
    pten::funcs::ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        dev_ctx, x_dim, y_dim, dout, dout, out, dout, axis, dx, dy, dx_op,
        dy_op);
  } else {
    pten::ElemwiseGradComputeWithBroadcast<T, DX_OP, DY_OP>(
        dev_ctx, x_dim, y_dim, dout, dout, out, dout, axis, dx, dy, dx_op,
        dy_op);
  }
}

// It is a common implementation to compute binary calculation with the support
// of broadcast, supporting both CPU and GPU.
// - CPU implementation cannot support the case when x needs broadcast, thus
//   this function need to be called with XxxFunctor and XxxInverseFunctor,
//   like AddFunctor and InverseAddFunctor.
// - GPU implementation supports all the broadcast cases, thus there is no need
//   to define and call with XxxInverseFunctor.
// TODO(liuyiqun): optimize the CPU implementation to support all broadcast
// cases and avoid the need of XxxInverseFunctor.
template <typename Functor, typename DeviceContext, typename T,
          typename OutType = T>
void ElementwiseComputeEx(const framework::ExecutionContext &ctx,
                          const framework::Tensor *x,
                          const framework::Tensor *y, int axis, Functor func,
                          framework::Tensor *z) {
  if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
    std::vector<const framework::Tensor *> ins = {x, y};
    std::vector<framework::Tensor *> outs = {z};
    z->mutable_data<OutType>(ctx.GetPlace());

    const auto &dev_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T,
                                                   OutType>(dev_ctx, ins, &outs,
                                                            axis, func);
#endif
    return;
  }

  z->mutable_data<OutType>(ctx.GetPlace());
  auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
  auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);
  auto pt_z = paddle::experimental::MakePtenDenseTensor(*z);

  const auto &dev_ctx =
      ctx.template device_context<platform::CPUDeviceContext>();
  pten::ElementwiseCompute<Functor, T, OutType>(
      dev_ctx, *pt_x.get(), *pt_y.get(), axis, func, pt_z.get());
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

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActBroadcast1CUDAKernel(
    const T *x, const T *y, int h, int w, CompoundFunctor compound_functor,
    T *out, T *intermediate_out) {
  int i = blockIdx.x;
  int j = threadIdx.x;

  while (j < w) {
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

    j += ELEMWISE_MAX_BLOCK_DIM;
  }
}

template <typename T, typename CompoundFunctor, bool BcastY,
          bool KeepIntermediateOut, bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast1CUDA(gpuStream_t stream, const T *x,
                                              const T *y,
                                              CompoundFunctor compound_functor,
                                              int h, int w, T *out,
                                              T *intermediate_out) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, w);
  int gird_size = h;
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
static void FusedElemwiseAndActBroadcast2CUDA(gpuStream_t stream, const T *x,
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

  int pre, n, post, is_run_common_broadcast;
  pten::funcs::get_mid_dims(x_dim, y_dim, axis, &pre, &n, &post,
                            &is_run_common_broadcast);
  if (post == 1) {
    int h = pre;
    int w = n;
    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
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
#if defined(__NVCC__) || defined(__HIPCC__)
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
template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut>
struct FusedElemwiseAndActGradNoBroadcast {
  HOSTDEVICE void operator()(size_t i) {
    T zero = static_cast<T>(0);
    T x_val = (x_ == nullptr) ? zero : x_[i];
    T y_val = (y_ == nullptr) ? zero : y_[i];
    T out_val = out_[i];
    T dout_val = dout_[i];
    T intermediate_out_val = UseIntermediateOut
                                 ? intermediate_out_[i]
                                 : dx_op_.GetIntermediateOut(x_val, y_val);
    if (dx_ != nullptr) {
      dx_[i] = dx_op_.UseIntermediateOut(x_val, y_val, intermediate_out_val,
                                         out_val, dout_val);
    }
    if (dy_ != nullptr) {
      dy_[i] = dy_op_.UseIntermediateOut(x_val, y_val, intermediate_out_val,
                                         out_val, dout_val);
    }
    if (dintermediate_ != nullptr) {
      dintermediate_[i] = dintermediate_op_.UseIntermediateOut(
          x_val, intermediate_out_val, out_val, dout_val);
    }
  }

  const T *x_;
  const T *y_;
  const T *intermediate_out_;
  const T *out_;
  const T *dout_;
  DX_OP dx_op_;
  DY_OP dy_op_;
  DIntermediate_OP dintermediate_op_;
  T *dx_;
  T *dy_;
  T *dintermediate_;
};

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          typename DIntermediate_OP, bool UseIntermediateOut>
void FusedElemwiseAndActGradComputeNoBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim, const framework::Tensor *x,
    const framework::Tensor *y, const framework::Tensor *intermediate_out,
    const framework::Tensor *out, const framework::Tensor *dout, int axis,
    framework::Tensor *dx, framework::Tensor *dy,
    framework::Tensor *dintermediate, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op) {
  size_t N = static_cast<size_t>(framework::product(x_dim));
  platform::ForRange<DeviceContext> for_range(
      ctx.template device_context<DeviceContext>(), N);
  const T *x_data = nullptr;
  const T *y_data = nullptr;
  if (x->IsInitialized()) x_data = x->data<T>();
  if (y->IsInitialized()) y_data = y->data<T>();

  for_range(FusedElemwiseAndActGradNoBroadcast<
            T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut>{
      x_data, y_data, intermediate_out ? intermediate_out->data<T>() : nullptr,
      out->data<T>(), dout->data<T>(), dx_op, dy_op, dintermediate_op,
      dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
      dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
      dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                               ctx.GetPlace())});
}

template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast1CPU(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int h, int w, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op, T *dx, T *dy, T *d_intermediate) {
  int64_t tmp_out_idx, x_idx, y_idx;
  T zero = static_cast<T>(0);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int offset = i * w + j;

      tmp_out_idx = BcastY ? j : offset;
      y_idx = BcastY ? j : offset;
      x_idx = BcastY ? offset : j;
      T x_val = (x == nullptr) ? zero : x[x_idx];
      T y_val = (y == nullptr) ? zero : y[y_idx];

      if (SameShapeOfIntermediateOutAndOut) {
        tmp_out_idx = offset;
      }

      if (dx != nullptr) {
        T tmp = UseIntermediateOut
                    ? dx_op.UseIntermediateOut(x_val, y_val,
                                               intermediate_out[tmp_out_idx],
                                               out[offset], dout[offset])
                    : dx_op.Recompute(x_val, y_val, out[offset], dout[offset]);

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
                    ? dy_op.UseIntermediateOut(x_val, y_val,
                                               intermediate_out[tmp_out_idx],
                                               out[offset], dout[offset])
                    : dy_op.Recompute(x_val, y_val, out[offset], dout[offset]);
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
      if (d_intermediate != nullptr) {
        T tmp = UseIntermediateOut
                    ? dintermediate_op.UseIntermediateOut(
                          x_val, intermediate_out[tmp_out_idx], out[offset],
                          dout[offset])
                    : dintermediate_op.Recompute(x_val, y_val, out[offset],
                                                 dout[i]);
        if (SameShapeOfIntermediateOutAndOut) {
          d_intermediate[tmp_out_idx] = tmp;
        } else {
          if (i == 0) {
            d_intermediate[tmp_out_idx] = tmp;
          } else {
            d_intermediate[tmp_out_idx] += tmp;
          }
        }
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast2CPU(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int pre, int n, int post, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op, T *dx, T *dy, T *d_intermediate) {
  int64_t tmp_out_idx, x_idx, y_idx;
  T zero = static_cast<T>(0);
  for (int i = 0; i < pre; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < post; ++k) {
        int offset = i * n * post + j * post + k;

        tmp_out_idx = BcastY ? j : offset;
        y_idx = BcastY ? j : offset;
        x_idx = BcastY ? offset : j;

        T x_val = (x == nullptr) ? zero : x[x_idx];
        T y_val = (y == nullptr) ? zero : y[y_idx];

        if (SameShapeOfIntermediateOutAndOut) {
          tmp_out_idx = offset;
        }

        if (dx != nullptr) {
          T tmp =
              UseIntermediateOut
                  ? dx_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dx_op.Recompute(x_val, y_val, out[offset], dout[offset]);

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
          T tmp =
              UseIntermediateOut
                  ? dy_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dy_op.Recompute(x_val, y_val, out[offset], dout[offset]);
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
        if (d_intermediate != nullptr) {
          T tmp = UseIntermediateOut
                      ? dintermediate_op.UseIntermediateOut(
                            x_val, intermediate_out[tmp_out_idx], out[offset],
                            dout[offset])
                      : dintermediate_op.Recompute(x_val, y_val, out[offset],
                                                   dout[i]);
          if (SameShapeOfIntermediateOutAndOut) {
            d_intermediate[tmp_out_idx] = tmp;
          } else {
            if (i == 0) {
              d_intermediate[tmp_out_idx] = tmp;
            } else {
              d_intermediate[tmp_out_idx] += tmp;
            }
          }
        }
      }
    }
  }
}

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActGradBroadcast1CUDAKernel(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int h, int w, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op, T *dx, T *dy, T *d_intermediate) {
  __shared__ T sdata[BLOCK_Y][BLOCK_X];
  size_t idx = threadIdx.x + BLOCK_X * blockIdx.x;
  size_t width_stride = gridDim.x * BLOCK_X;

  size_t full_w = ROUNDUP(w, BLOCK_X);

  T zero = static_cast<T>(0);

  for (size_t j = idx; j < full_w; j += width_stride) {
    T val(0), inter_val(0);
    if (j < w) {
      for (size_t i = threadIdx.y; i < h; i += BLOCK_Y) {
        size_t offset = i * w + j;

        size_t tmp_out_idx = BcastY ? j : offset;
        size_t y_idx = BcastY ? j : offset;
        size_t x_idx = BcastY ? offset : j;
        T x_val = (x == nullptr) ? zero : x[x_idx];
        T y_val = (y == nullptr) ? zero : y[y_idx];

        if (SameShapeOfIntermediateOutAndOut) {
          tmp_out_idx = offset;
        }

        if (dx != nullptr) {
          T tmp =
              UseIntermediateOut
                  ? dx_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dx_op.Recompute(x_val, y_val, out[offset], dout[offset]);

          if (BcastY) {
            dx[x_idx] = tmp;
          } else {
            val += tmp;
          }
        }
        if (dy != nullptr) {
          T tmp =
              UseIntermediateOut
                  ? dy_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dy_op.Recompute(x_val, y_val, out[offset], dout[offset]);
          if (BcastY) {
            val += tmp;
          } else {
            dy[y_idx] = tmp;
          }
        }
        if (d_intermediate != nullptr) {
          T tmp = UseIntermediateOut
                      ? dintermediate_op.UseIntermediateOut(
                            y[y_idx], intermediate_out[tmp_out_idx],
                            out[offset], dout[offset])
                      : dintermediate_op.Recompute(x_val, y_val, out[offset],
                                                   dout[offset]);
          if (SameShapeOfIntermediateOutAndOut) {
            d_intermediate[tmp_out_idx] = tmp;
          } else {
            inter_val += tmp;
          }
        }
      }
    }

    // transpose, for ReduceSum with wrap
    sdata[threadIdx.y][threadIdx.x] = val;
    __syncthreads();
    val = sdata[threadIdx.x][threadIdx.y];
#pragma unroll
    for (int i = BLOCK_X >> 1; i > 0; i >>= 1) {
      // reduce sum with wrap
      val += platform::CudaShuffleXorSync(0xFFFFFFFF, val, i);
    }

    size_t idx_j = j + threadIdx.y;
    if (BcastY) {
      if (dy) {
        if (threadIdx.x == 0 && (idx_j < w)) dy[idx_j] = val;
      }
    } else {
      if (dx) {
        if (threadIdx.x == 0 && (idx_j < w)) dx[idx_j] = val;
      }
    }

    if (!SameShapeOfIntermediateOutAndOut) {
      if (d_intermediate) {
        sdata[threadIdx.y][threadIdx.x] = inter_val;
        __syncthreads();
        inter_val = sdata[threadIdx.x][threadIdx.y];
#pragma unroll
        for (int i = BLOCK_X >> 1; i > 0; i >>= 1) {
          // reduce sum with wrap
          inter_val += platform::CudaShuffleXorSync(0xFFFFFFFF, inter_val, i);
        }
        if (threadIdx.x == 0 && (idx_j < w)) d_intermediate[idx_j] = inter_val;
      }
    }
  }  // end for
}

template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast1CUDA(
    const framework::ExecutionContext &ctx, const T *x, const T *y,
    const T *intermediate_out, const T *out, const T *dout, int h, int w,
    DX_OP dx_op, DY_OP dy_op, DIntermediate_OP dintermediate_op, T *dx, T *dy,
    T *d_intermediate) {
  gpuStream_t stream = ctx.cuda_device_context().stream();

  dim3 blocks(BLOCK_X, BLOCK_Y);
  int max_gpu_threads = ctx.cuda_device_context().GetMaxPhysicalThreadCount();
  int max_blocks = std::max(max_gpu_threads / (BLOCK_X * BLOCK_Y), 1);
  int theory_block = (w + BLOCK_X - 1) / BLOCK_X;
  dim3 grids(std::min(theory_block, max_blocks));

  FusedElemwiseAndActGradBroadcast1CUDAKernel<
      T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut, BcastY,
      SameShapeOfIntermediateOutAndOut><<<grids, blocks, 0, stream>>>(
      x, y, intermediate_out, out, dout, h, w, dx_op, dy_op, dintermediate_op,
      dx, dy, d_intermediate);
}

template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActGradBroadcast2CUDAKernel(
    const T *x, const T *y, const T *intermediate_out, const T *out,
    const T *dout, int pre, int n, int post, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op, T *dx, T *dy, T *d_intermediate) {
  int tid = threadIdx.x;
  int j = blockIdx.x;

  T val(0), inter_val(0);
  int ttid = tid;
  int64_t tmp_out_idx, x_idx, y_idx;
  T zero = static_cast<T>(0);
  while (true) {
    int i = ttid / post;
    int k = ttid % post;
    if (i >= pre) break;

    int offset = i * n * post + j * post + k;

    tmp_out_idx = BcastY ? j : offset;
    y_idx = BcastY ? j : offset;
    x_idx = BcastY ? offset : j;
    T x_val = (x == nullptr) ? zero : x[x_idx];
    T y_val = (y == nullptr) ? zero : y[y_idx];

    if (SameShapeOfIntermediateOutAndOut) {
      tmp_out_idx = offset;
    }

    if (dx != nullptr) {
      T tmp = UseIntermediateOut
                  ? dx_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dx_op.Recompute(x_val, y_val, out[offset], dout[offset]);

      if (BcastY) {
        dx[x_idx] = tmp;
      } else {
        val += tmp;
      }
    }
    if (dy != nullptr) {
      T tmp = UseIntermediateOut
                  ? dy_op.UseIntermediateOut(x_val, y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset], dout[offset])
                  : dy_op.Recompute(x_val, y_val, out[offset], dout[offset]);
      if (BcastY) {
        val += tmp;
      } else {
        dy[y_idx] = tmp;
      }
    }
    if (d_intermediate != nullptr) {
      T tmp = UseIntermediateOut
                  ? dintermediate_op.UseIntermediateOut(
                        y_val, intermediate_out[tmp_out_idx], out[offset],
                        dout[offset])
                  : dintermediate_op.Recompute(x_val, y_val, out[offset],
                                               dout[offset]);
      if (SameShapeOfIntermediateOutAndOut) {
        d_intermediate[tmp_out_idx] = tmp;
      } else {
        inter_val += tmp;
      }
    }
    ttid += ELEMWISE_MAX_BLOCK_DIM;
  }

  int h = pre * post;
  h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
  if (BcastY) {
    if (dy) {
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {
    if (dx) {
      val = paddle::platform::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
  if (!SameShapeOfIntermediateOutAndOut) {
    if (d_intermediate) {
      inter_val = paddle::platform::reduceSum(inter_val, tid, h);
      if (threadIdx.x == 0) {
        d_intermediate[j] = inter_val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename DIntermediate_OP,
          bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast2CUDA(
    gpuStream_t stream, const T *x, const T *y, const T *intermediate_out,
    const T *out, const T *dout, int pre, int n, int post, DX_OP dx_op,
    DY_OP dy_op, DIntermediate_OP dintermediate_op, T *dx, T *dy,
    T *dintermediate) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;
  FusedElemwiseAndActGradBroadcast2CUDAKernel<
      T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut, BcastY,
      SameShapeOfIntermediateOutAndOut><<<gird_size, block_size, 0, stream>>>(
      x, y, intermediate_out, out, dout, pre, n, post, dx_op, dy_op,
      dintermediate_op, dx, dy, dintermediate);
}
#endif

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          typename DIntermediate_OP, bool UseIntermediateOut, bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActGradComputeWithBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dim,
    const framework::DDim &y_dim_untrimed, const framework::Tensor *x,
    const framework::Tensor *y, const framework::Tensor *intermediate_out,
    const framework::Tensor *out, const framework::Tensor *dout, int axis,
    framework::Tensor *dx, framework::Tensor *dy,
    framework::Tensor *dintermediate, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op) {
  axis = (axis == -1 ? x_dim.size() - y_dim_untrimed.size() : axis);
  auto y_dim = trim_trailing_singular_dims(y_dim_untrimed);
  axis = (y_dim.size() == 0) ? x_dim.size() : axis;

  int pre, n, post, is_run_common_broadcast;
  pten::funcs::get_mid_dims(x_dim, y_dim, axis, &pre, &n, &post,
                            &is_run_common_broadcast);
  const T *x_data = nullptr;
  const T *y_data = nullptr;
  if (x->IsInitialized()) x_data = x->data<T>();
  if (y->IsInitialized()) y_data = y->data<T>();
  if (post == 1) {
    int h = pre;
    int w = n;

    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      FusedElemwiseAndActGradBroadcast1CUDA<T, DX_OP, DY_OP, DIntermediate_OP,
                                            UseIntermediateOut, BcastY,
                                            SameShapeOfIntermediateOutAndOut>(
          ctx, x_data, y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), h, w, dx_op, dy_op, dintermediate_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
          dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                                   ctx.GetPlace()));
#endif
    } else {
      FusedElemwiseAndActGradBroadcast1CPU<T, DX_OP, DY_OP, DIntermediate_OP,
                                           UseIntermediateOut, BcastY,
                                           SameShapeOfIntermediateOutAndOut>(
          x_data, y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), h, w, dx_op, dy_op, dintermediate_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
          dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                                   ctx.GetPlace()));
    }
  } else {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      FusedElemwiseAndActGradBroadcast2CUDA<T, DX_OP, DY_OP, DIntermediate_OP,
                                            UseIntermediateOut, BcastY,
                                            SameShapeOfIntermediateOutAndOut>(
          ctx.template device_context<DeviceContext>().stream(), x_data, y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), pre, n, post, dx_op, dy_op,
          dintermediate_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
          dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                                   ctx.GetPlace()));
#endif
    } else {
      FusedElemwiseAndActGradBroadcast2CPU<T, DX_OP, DY_OP, DIntermediate_OP,
                                           UseIntermediateOut, BcastY,
                                           SameShapeOfIntermediateOutAndOut>(
          x_data, y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(), dout->data<T>(), pre, n, post, dx_op, dy_op,
          dintermediate_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()),
          dintermediate == nullptr ? nullptr : dintermediate->mutable_data<T>(
                                                   ctx.GetPlace()));
    }
  }
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          typename DIntermediate_OP, bool UseIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActGradComputeEx(
    const framework::ExecutionContext &ctx, const framework::Tensor *x,
    const framework::Tensor *y, const framework::Tensor *out,
    const framework::Tensor *intermediate_out, const framework::Tensor *dout,
    int axis, framework::Tensor *dx, framework::Tensor *dy,
    framework::Tensor *dintermediate, DX_OP dx_op, DY_OP dy_op,
    DIntermediate_OP dintermediate_op) {
  const framework::DDim &x_dim = x->dims();
  const framework::DDim &y_dim = y->dims();
  if (UseIntermediateOut) {
    PADDLE_ENFORCE_NOT_NULL(
        intermediate_out,
        platform::errors::InvalidArgument("Intermediate out is null pointer."));
  }
  if (x_dim == y_dim) {
    FusedElemwiseAndActGradComputeNoBroadcast<
        DeviceContext, T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut>(
        ctx, x_dim, y_dim, x, y, intermediate_out, out, dout, axis, dx, dy,
        dintermediate, dx_op, dy_op, dintermediate_op);
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
          DeviceContext, T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut,
          true /*BcastY*/, SameShapeOfIntermediateOutAndOut>(
          ctx, x_dim, y_dim, x, y, intermediate_out, out, dout, axis, dx, dy,
          dintermediate, dx_op, dy_op, dintermediate_op);
    } else {
      FusedElemwiseAndActGradComputeWithBroadcast<
          DeviceContext, T, DX_OP, DY_OP, DIntermediate_OP, UseIntermediateOut,
          false /*BcastY*/, SameShapeOfIntermediateOutAndOut>(
          ctx, y_dim, x_dim, x, y, intermediate_out, out, dout, axis, dx, dy,
          dintermediate, dx_op, dy_op, dintermediate_op);
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
    PADDLE_ENFORCE_NOT_NULL(
        intermediate_out,
        platform::errors::InvalidArgument(
            "The save_intermediate_out is opened, intermediate "
            "out is null pointer."));
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
    bool bcast_y = x.numel() >= y.numel();
    // z = f1(x, f2(y))
    // z = f1(f2(x, y))
    if (bcast_y) {  // Y should be broadcast.
      // In this case,
      // for 'f2(y)', the shape of intermediate_out should be equal to the
      // shape
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
      // for 'f2(y)', the shape of intermediate_out should be equal to the
      // shape
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

template <typename DeviceContext, typename T>
static inline void GetDoubleGradSafeTensor(
    const framework::ExecutionContext &ctx, const framework::Tensor *x,
    const framework::Tensor *ddx, framework::Tensor *ddx_safe) {
  if (ddx) {
    *ddx_safe = *ddx;
  } else {
    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    *ddx_safe = ctx.AllocateTmpTensor<T, DeviceContext>(x->dims(), dev_ctx);
    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(ctx.template device_context<DeviceContext>(), ddx_safe,
             static_cast<T>(0));
  }
}

// for broadcast backwards
static inline std::vector<int> GetReduceDim(const framework::DDim &in,
                                            const framework::DDim &out,
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

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
void ReduceWrapper(const platform::CUDADeviceContext &dev_ctx, int axis,
                   framework::Tensor *src, framework::Tensor *dst) {
  std::vector<int> reduce_dims = GetReduceDim(dst->dims(), src->dims(), axis);
  TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
      *src, dst, kps::IdentityFunctor<T>(), reduce_dims, dev_ctx.stream());
}

template <ElementwiseType ET, typename T, typename Functor>
void GetGradXAndYOut(const platform::CUDADeviceContext &dev_ctx,
                     const platform::Place &place, int axis,
                     std::vector<const framework::Tensor *> ins,
                     const framework::Tensor *dout, framework::Tensor *dx,
                     framework::Tensor *dy, Functor func) {
  framework::Tensor tmp_dx;
  framework::Tensor tmp_dy;
  dx->mutable_data<T>(place);
  dy->mutable_data<T>(place);
  std::vector<framework::Tensor *> outs;
  if (dx->dims() == dout->dims() && dy->dims() == dout->dims()) {
    outs = {dx, dy};
  } else if (dx->dims() != dout->dims() && dy->dims() == dout->dims()) {
    tmp_dx.mutable_data<T>(dout->dims(), place);
    outs = {&tmp_dx, dy};
  } else if (dx->dims() == dout->dims() && dy->dims() != dout->dims()) {
    tmp_dy.mutable_data<T>(dout->dims(), place);
    outs = {dx, &tmp_dy};
  } else if (dx->dims() != dout->dims() && dy->dims() != dout->dims()) {
    tmp_dy.mutable_data<T>(dout->dims(), place);
    tmp_dx.mutable_data<T>(dout->dims(), place);
    outs = {&tmp_dx, &tmp_dy};
  }

  paddle::operators::LaunchElementwiseCudaKernel<ET, T, T, decltype(func), 2>(
      dev_ctx, ins, &outs, axis, func);

  if (dx->dims() != dout->dims() && dy->dims() == dout->dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dx, dx);
  } else if (dx->dims() == dout->dims() && dy->dims() != dout->dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dy, dy);
  } else if (dx->dims() != dout->dims() && dy->dims() != dout->dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dx, dx);
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dy, dy);
  }
}

template <ElementwiseType ET, typename T, typename Functor>
void GetGradXOrYOut(const platform::CUDADeviceContext &dev_ctx,
                    const platform::Place &place, int axis,
                    std::vector<const framework::Tensor *> ins,
                    const framework::Tensor *dout, framework::Tensor *dxy,
                    Functor func) {
  framework::Tensor tmp_dxy;
  dxy->mutable_data<T>(place);

  std::vector<framework::Tensor *> outs;
  if (dxy->dims() != dout->dims()) {
    tmp_dxy.mutable_data<T>(dout->dims(), place);
    outs = {&tmp_dxy};
  } else {
    outs = {dxy};
  }

  paddle::operators::LaunchElementwiseCudaKernel<ET, T, T>(dev_ctx, ins, &outs,
                                                           axis, func);
  if (dxy->dims() != dout->dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dxy, dxy);
  }
}

#endif

}  // namespace operators
}  // namespace paddle
