// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <functional>  // for multiplies
#include <iterator>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/cpu/elementwise_grad.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#ifdef __NVCC__
#include <cuda.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#include <thrust/iterator/iterator_adaptor.h>

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/gpu/elementwise_grad.h"

#endif

#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

namespace phi {
namespace funcs {

// It is a common implementation to compute binary calculation with the support
// of broadcast, supporting both CPU and GPU.
// - CPU implementation cannot support the case when x needs broadcast, thus
//   this function need to be called with XxxFunctor and XxxInverseFunctor,
//   like AddFunctor and InverseAddFunctor.
// - GPU implementation supports all the broadcast cases, thus there is no need
//   to define and call with XxxInverseFunctor.
// TODO(liuyiqun): optimize the CPU implementation to support all broadcast
// cases and avoid the need of XxxInverseFunctor.
template <typename Functor,
          typename DeviceContext,
          typename T,
          typename OutType = T>
void ElementwiseComputeEx(const DeviceContext &dev_ctx,
                          const phi::DenseTensor *x,
                          const phi::DenseTensor *y,
                          int axis,
                          Functor func,
                          phi::DenseTensor *z) {
  dev_ctx.template Alloc<OutType>(z);
  phi::funcs::ElementwiseCompute<Functor, T, OutType>(
      dev_ctx, *x, *y, func, z, axis);
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
template <typename T,
          typename CompoundFunctor,
          bool BcastY,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast1CPU(const T *x,
                                             const T *y,
                                             CompoundFunctor compound_functor,
                                             int h,
                                             int w,
                                             T *out,
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
template <typename T,
          typename CompoundFunctor,
          bool BcastY,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast2CPU(const T *x,
                                             const T *y,
                                             int pre,
                                             int n,
                                             int post,
                                             CompoundFunctor compound_functor,
                                             T *out,
                                             T *intermediate_out) {
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
template <typename T,
          typename CompoundFunctor,
          bool BcastY,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActBroadcast1CUDAKernel(
    const T *x,
    const T *y,
    int h,
    int w,
    CompoundFunctor compound_functor,
    T *out,
    T *intermediate_out) {
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

template <typename T,
          typename CompoundFunctor,
          bool BcastY,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast1CUDA(gpuStream_t stream,
                                              const T *x,
                                              const T *y,
                                              CompoundFunctor compound_functor,
                                              int h,
                                              int w,
                                              T *out,
                                              T *intermediate_out) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, w);
  int gird_size = h;
  FusedElemwiseAndActBroadcast1CUDAKernel<T,
                                          CompoundFunctor,
                                          BcastY,
                                          KeepIntermediateOut,
                                          SameShapeOfIntermediateOutAndOut>
      <<<gird_size, block_size, 0, stream>>>(
          x, y, h, w, compound_functor, out, intermediate_out);
}

template <typename T,
          typename CompoundFunctor,
          bool BcastY,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActBroadcast2CUDAKernel(
    const T *x,
    const T *y,
    CompoundFunctor compound_functor,
    int pre,
    int n,
    int post,
    T *out,
    T *intermediate_out) {
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

template <typename T,
          typename CompoundFunctor,
          bool BcastY,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActBroadcast2CUDA(gpuStream_t stream,
                                              const T *x,
                                              const T *y,
                                              int pre,
                                              int n,
                                              int post,
                                              CompoundFunctor compound_functor,
                                              T *out,
                                              T *intermediate_out) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;

  FusedElemwiseAndActBroadcast2CUDAKernel<T,
                                          CompoundFunctor,
                                          BcastY,
                                          KeepIntermediateOut,
                                          SameShapeOfIntermediateOutAndOut>
      <<<gird_size, block_size, 0, stream>>>(
          x, y, compound_functor, pre, n, post, out, intermediate_out);
}

#endif

template <typename DeviceContext,
          typename T,
          typename CompoundFunctor,
          bool KeepIntermediateOut>
void FusedElemwiseAndActComputeNoBroadcast(const DeviceContext &dev_ctx,
                                           const phi::DDim &x_dim,
                                           const phi::DenseTensor &x,
                                           const phi::DenseTensor &y,
                                           CompoundFunctor compound_functor,
                                           phi::DenseTensor *out,
                                           phi::DenseTensor *intermediate_out) {
  size_t N = static_cast<size_t>(common::product(x_dim));

  phi::funcs::ForRange<DeviceContext> for_range(dev_ctx, N);

  for_range(
      FusedElemwiseAndActNoBroadcast<T, CompoundFunctor, KeepIntermediateOut>{
          x.data<T>(),
          y.data<T>(),
          compound_functor,
          dev_ctx.template Alloc<T>(out),
          intermediate_out == nullptr
              ? nullptr
              : dev_ctx.template Alloc<T>(intermediate_out)});
}

template <typename DeviceContext,
          typename T,
          typename CompoundFunctor,
          bool BcastY,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActComputeWithBroadcast(
    const DeviceContext &dev_ctx,
    const phi::DDim &x_dim,
    const phi::DDim &y_dim_untrimed,
    const phi::DenseTensor &x,
    const phi::DenseTensor &y,
    CompoundFunctor compound_functor,
    int axis,
    phi::DenseTensor *out,
    phi::DenseTensor *intermediate_out) {
  axis = (axis == -1 ? x_dim.size() - y_dim_untrimed.size() : axis);
  auto y_dim = phi::funcs::TrimTrailingSingularDims(y_dim_untrimed);
  axis = (y_dim.size() == 0) ? x_dim.size() : axis;

  int pre, n, post, is_run_common_broadcast;
  phi::funcs::GetMidDims(
      x_dim, y_dim, axis, &pre, &n, &post, &is_run_common_broadcast);
  if (post == 1) {
    int h = pre;
    int w = n;
    if (dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU) {
#if defined(__NVCC__) || defined(__HIPCC__)
      FusedElemwiseAndActBroadcast1CUDA<T,
                                        CompoundFunctor,
                                        BcastY,
                                        KeepIntermediateOut,
                                        SameShapeOfIntermediateOutAndOut>(
          dev_ctx.stream(),
          x.data<T>(),
          y.data<T>(),
          compound_functor,
          h,
          w,
          dev_ctx.template Alloc<T>(out),
          intermediate_out == nullptr
              ? nullptr
              : dev_ctx.template Alloc<T>(intermediate_out));
#endif
    } else {
      FusedElemwiseAndActBroadcast1CPU<T,
                                       CompoundFunctor,
                                       BcastY,
                                       KeepIntermediateOut,
                                       SameShapeOfIntermediateOutAndOut>(
          x.data<T>(),
          y.data<T>(),
          compound_functor,
          h,
          w,
          dev_ctx.template Alloc<T>(out),
          intermediate_out == nullptr
              ? nullptr
              : dev_ctx.template Alloc<T>(intermediate_out));
    }
  } else {
    if (dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU) {
#if defined(__NVCC__) || defined(__HIPCC__)
      FusedElemwiseAndActBroadcast2CUDA<T,
                                        CompoundFunctor,
                                        BcastY,
                                        KeepIntermediateOut,
                                        SameShapeOfIntermediateOutAndOut>(
          dev_ctx.stream(),
          x.data<T>(),
          y.data<T>(),
          pre,
          n,
          post,
          compound_functor,
          dev_ctx.template Alloc<T>(out),
          intermediate_out == nullptr
              ? nullptr
              : dev_ctx.template Alloc<T>(intermediate_out));
#endif
    } else {
      FusedElemwiseAndActBroadcast2CPU<T,
                                       CompoundFunctor,
                                       BcastY,
                                       KeepIntermediateOut,
                                       SameShapeOfIntermediateOutAndOut>(
          x.data<T>(),
          y.data<T>(),
          pre,
          n,
          post,
          compound_functor,
          dev_ctx.template Alloc<T>(out),
          intermediate_out == nullptr
              ? nullptr
              : dev_ctx.template Alloc<T>(intermediate_out));
    }
  }
}

// --- backward
template <typename T,
          typename DX_OP,
          typename DY_OP,
          typename DIntermediate_OP,
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
      dx_[i] = dx_op_.UseIntermediateOut(
          x_val, y_val, intermediate_out_val, out_val, dout_val);
    }
    if (dy_ != nullptr) {
      dy_[i] = dy_op_.UseIntermediateOut(
          x_val, y_val, intermediate_out_val, out_val, dout_val);
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

template <typename DeviceContext,
          typename T,
          typename DX_OP,
          typename DY_OP,
          typename DIntermediate_OP,
          bool UseIntermediateOut>
void FusedElemwiseAndActGradComputeNoBroadcast(
    const DeviceContext &dev_ctx,
    const phi::DDim &x_dim,
    const phi::DDim &y_dim UNUSED,
    const phi::DenseTensor *x,
    const phi::DenseTensor *y,
    const phi::DenseTensor *intermediate_out,
    const phi::DenseTensor *out,
    const phi::DenseTensor *dout,
    int axis UNUSED,
    phi::DenseTensor *dx,
    phi::DenseTensor *dy,
    phi::DenseTensor *dintermediate,
    DX_OP dx_op,
    DY_OP dy_op,
    DIntermediate_OP dintermediate_op) {
  size_t N = static_cast<size_t>(common::product(x_dim));
  phi::funcs::ForRange<DeviceContext> for_range(dev_ctx, N);
  const T *x_data = nullptr;
  const T *y_data = nullptr;
  if (x->IsInitialized()) x_data = x->data<T>();
  if (y->IsInitialized()) y_data = y->data<T>();

  for_range(FusedElemwiseAndActGradNoBroadcast<T,
                                               DX_OP,
                                               DY_OP,
                                               DIntermediate_OP,
                                               UseIntermediateOut>{
      x_data,
      y_data,
      intermediate_out ? intermediate_out->data<T>() : nullptr,
      out->data<T>(),
      dout->data<T>(),
      dx_op,
      dy_op,
      dintermediate_op,
      dx == nullptr ? nullptr : dev_ctx.template Alloc<T>(dx),
      dy == nullptr ? nullptr : dev_ctx.template Alloc<T>(dy),
      dintermediate == nullptr ? nullptr
                               : dev_ctx.template Alloc<T>(dintermediate)});
}

template <typename T,
          typename DX_OP,
          typename DY_OP,
          typename DIntermediate_OP,
          bool UseIntermediateOut,
          bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast1CPU(
    const T *x,
    const T *y,
    const T *intermediate_out,
    const T *out,
    const T *dout,
    int h,
    int w,
    DX_OP dx_op,
    DY_OP dy_op,
    DIntermediate_OP dintermediate_op,
    T *dx,
    T *dy,
    T *d_intermediate) {
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
                    ? dx_op.UseIntermediateOut(x_val,
                                               y_val,
                                               intermediate_out[tmp_out_idx],
                                               out[offset],
                                               dout[offset])
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
                    ? dy_op.UseIntermediateOut(x_val,
                                               y_val,
                                               intermediate_out[tmp_out_idx],
                                               out[offset],
                                               dout[offset])
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
        T tmp = UseIntermediateOut ? dintermediate_op.UseIntermediateOut(
                                         x_val,
                                         intermediate_out[tmp_out_idx],
                                         out[offset],
                                         dout[offset])
                                   : dintermediate_op.Recompute(
                                         x_val, y_val, out[offset], dout[i]);
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

template <typename T,
          typename DX_OP,
          typename DY_OP,
          typename DIntermediate_OP,
          bool UseIntermediateOut,
          bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast2CPU(
    const T *x,
    const T *y,
    const T *intermediate_out,
    const T *out,
    const T *dout,
    int pre,
    int n,
    int post,
    DX_OP dx_op,
    DY_OP dy_op,
    DIntermediate_OP dintermediate_op,
    T *dx,
    T *dy,
    T *d_intermediate) {
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
                  ? dx_op.UseIntermediateOut(x_val,
                                             y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset],
                                             dout[offset])
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
                  ? dy_op.UseIntermediateOut(x_val,
                                             y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset],
                                             dout[offset])
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
          T tmp = UseIntermediateOut ? dintermediate_op.UseIntermediateOut(
                                           x_val,
                                           intermediate_out[tmp_out_idx],
                                           out[offset],
                                           dout[offset])
                                     : dintermediate_op.Recompute(
                                           x_val, y_val, out[offset], dout[i]);
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
template <typename T,
          typename DX_OP,
          typename DY_OP,
          typename DIntermediate_OP,
          bool UseIntermediateOut,
          bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActGradBroadcast1CUDAKernel(
    const T *x,
    const T *y,
    const T *intermediate_out,
    const T *out,
    const T *dout,
    int h,
    int w,
    DX_OP dx_op,
    DY_OP dy_op,
    DIntermediate_OP dintermediate_op,
    T *dx,
    T *dy,
    T *d_intermediate) {
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
                  ? dx_op.UseIntermediateOut(x_val,
                                             y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset],
                                             dout[offset])
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
                  ? dy_op.UseIntermediateOut(x_val,
                                             y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset],
                                             dout[offset])
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
                            y[y_idx],
                            intermediate_out[tmp_out_idx],
                            out[offset],
                            dout[offset])
                      : dintermediate_op.Recompute(
                            x_val, y_val, out[offset], dout[offset]);
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
      val += phi::backends::gpu::CudaShuffleXorSync(0xFFFFFFFF, val, i);
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
          inter_val +=
              phi::backends::gpu::CudaShuffleXorSync(0xFFFFFFFF, inter_val, i);
        }
        if (threadIdx.x == 0 && (idx_j < w)) d_intermediate[idx_j] = inter_val;
      }
    }
  }  // end for
}

template <typename T,
          typename DX_OP,
          typename DY_OP,
          typename DIntermediate_OP,
          bool UseIntermediateOut,
          bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast1CUDA(
    const phi::GPUContext &dev_ctx,
    const T *x,
    const T *y,
    const T *intermediate_out,
    const T *out,
    const T *dout,
    int h,
    int w,
    DX_OP dx_op,
    DY_OP dy_op,
    DIntermediate_OP dintermediate_op,
    T *dx,
    T *dy,
    T *d_intermediate) {
  gpuStream_t stream = dev_ctx.stream();

  dim3 blocks(BLOCK_X, BLOCK_Y);
  int max_gpu_threads = dev_ctx.GetMaxPhysicalThreadCount();
  int max_blocks = std::max(max_gpu_threads / (BLOCK_X * BLOCK_Y), 1);
  int theory_block = (w + BLOCK_X - 1) / BLOCK_X;
  dim3 grids(std::min(theory_block, max_blocks));

  FusedElemwiseAndActGradBroadcast1CUDAKernel<T,
                                              DX_OP,
                                              DY_OP,
                                              DIntermediate_OP,
                                              UseIntermediateOut,
                                              BcastY,
                                              SameShapeOfIntermediateOutAndOut>
      <<<grids, blocks, 0, stream>>>(x,
                                     y,
                                     intermediate_out,
                                     out,
                                     dout,
                                     h,
                                     w,
                                     dx_op,
                                     dy_op,
                                     dintermediate_op,
                                     dx,
                                     dy,
                                     d_intermediate);
}

template <typename T,
          typename DX_OP,
          typename DY_OP,
          typename DIntermediate_OP,
          bool UseIntermediateOut,
          bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static __global__ void FusedElemwiseAndActGradBroadcast2CUDAKernel(
    const T *x,
    const T *y,
    const T *intermediate_out,
    const T *out,
    const T *dout,
    int pre,
    int n,
    int post,
    DX_OP dx_op,
    DY_OP dy_op,
    DIntermediate_OP dintermediate_op,
    T *dx,
    T *dy,
    T *d_intermediate) {
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
                  ? dx_op.UseIntermediateOut(x_val,
                                             y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset],
                                             dout[offset])
                  : dx_op.Recompute(x_val, y_val, out[offset], dout[offset]);

      if (BcastY) {
        dx[x_idx] = tmp;
      } else {
        val += tmp;
      }
    }
    if (dy != nullptr) {
      T tmp = UseIntermediateOut
                  ? dy_op.UseIntermediateOut(x_val,
                                             y_val,
                                             intermediate_out[tmp_out_idx],
                                             out[offset],
                                             dout[offset])
                  : dy_op.Recompute(x_val, y_val, out[offset], dout[offset]);
      if (BcastY) {
        val += tmp;
      } else {
        dy[y_idx] = tmp;
      }
    }
    if (d_intermediate != nullptr) {
      T tmp = UseIntermediateOut ? dintermediate_op.UseIntermediateOut(
                                       y_val,
                                       intermediate_out[tmp_out_idx],
                                       out[offset],
                                       dout[offset])
                                 : dintermediate_op.Recompute(
                                       x_val, y_val, out[offset], dout[offset]);
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
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dy[j] = val;
      }
    }
  } else {
    if (dx) {
      val = phi::backends::gpu::reduceSum(val, tid, h);
      if (threadIdx.x == 0) {
        dx[j] = val;
      }
    }
  }
  if (!SameShapeOfIntermediateOutAndOut) {
    if (d_intermediate) {
      inter_val = phi::backends::gpu::reduceSum(inter_val, tid, h);
      if (threadIdx.x == 0) {
        d_intermediate[j] = inter_val;
      }
    }
  }
}

template <typename T,
          typename DX_OP,
          typename DY_OP,
          typename DIntermediate_OP,
          bool UseIntermediateOut,
          bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
static void FusedElemwiseAndActGradBroadcast2CUDA(
    gpuStream_t stream,
    const T *x,
    const T *y,
    const T *intermediate_out,
    const T *out,
    const T *dout,
    int pre,
    int n,
    int post,
    DX_OP dx_op,
    DY_OP dy_op,
    DIntermediate_OP dintermediate_op,
    T *dx,
    T *dy,
    T *dintermediate) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;
  FusedElemwiseAndActGradBroadcast2CUDAKernel<T,
                                              DX_OP,
                                              DY_OP,
                                              DIntermediate_OP,
                                              UseIntermediateOut,
                                              BcastY,
                                              SameShapeOfIntermediateOutAndOut>
      <<<gird_size, block_size, 0, stream>>>(x,
                                             y,
                                             intermediate_out,
                                             out,
                                             dout,
                                             pre,
                                             n,
                                             post,
                                             dx_op,
                                             dy_op,
                                             dintermediate_op,
                                             dx,
                                             dy,
                                             dintermediate);
}
#endif

template <typename DeviceContext,
          typename T,
          typename DX_OP,
          typename DY_OP,
          typename DIntermediate_OP,
          bool UseIntermediateOut,
          bool BcastY,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActGradComputeWithBroadcast(
    const DeviceContext &dev_ctx,
    const phi::DDim &x_dim,
    const phi::DDim &y_dim_untrimed,
    const phi::DenseTensor *x,
    const phi::DenseTensor *y,
    const phi::DenseTensor *intermediate_out,
    const phi::DenseTensor *out,
    const phi::DenseTensor *dout,
    int axis,
    phi::DenseTensor *dx,
    phi::DenseTensor *dy,
    phi::DenseTensor *dintermediate,
    DX_OP dx_op,
    DY_OP dy_op,
    DIntermediate_OP dintermediate_op) {
  axis = (axis == -1 ? x_dim.size() - y_dim_untrimed.size() : axis);
  auto y_dim = phi::funcs::TrimTrailingSingularDims(y_dim_untrimed);
  axis = (y_dim.size() == 0) ? x_dim.size() : axis;

  int pre, n, post, is_run_common_broadcast;
  phi::funcs::GetMidDims(
      x_dim, y_dim, axis, &pre, &n, &post, &is_run_common_broadcast);
  const T *x_data = nullptr;
  const T *y_data = nullptr;
  if (x->IsInitialized()) x_data = x->data<T>();
  if (y->IsInitialized()) y_data = y->data<T>();
  if (post == 1) {
    int h = pre;
    int w = n;

    if (dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU) {
#if defined(__NVCC__) || defined(__HIPCC__)
      FusedElemwiseAndActGradBroadcast1CUDA<T,
                                            DX_OP,
                                            DY_OP,
                                            DIntermediate_OP,
                                            UseIntermediateOut,
                                            BcastY,
                                            SameShapeOfIntermediateOutAndOut>(
          dev_ctx,
          x_data,
          y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(),
          dout->data<T>(),
          h,
          w,
          dx_op,
          dy_op,
          dintermediate_op,
          dx == nullptr ? nullptr : dev_ctx.template Alloc<T>(dx),
          dy == nullptr ? nullptr : dev_ctx.template Alloc<T>(dy),
          dintermediate == nullptr ? nullptr
                                   : dev_ctx.template Alloc<T>(dintermediate));
#endif
    } else {
      FusedElemwiseAndActGradBroadcast1CPU<T,
                                           DX_OP,
                                           DY_OP,
                                           DIntermediate_OP,
                                           UseIntermediateOut,
                                           BcastY,
                                           SameShapeOfIntermediateOutAndOut>(
          x_data,
          y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(),
          dout->data<T>(),
          h,
          w,
          dx_op,
          dy_op,
          dintermediate_op,
          dx == nullptr ? nullptr : dev_ctx.template Alloc<T>(dx),
          dy == nullptr ? nullptr : dev_ctx.template Alloc<T>(dy),
          dintermediate == nullptr ? nullptr
                                   : dev_ctx.template Alloc<T>(dintermediate));
    }
  } else {
    if (dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU) {
#if defined(__NVCC__) || defined(__HIPCC__)
      FusedElemwiseAndActGradBroadcast2CUDA<T,
                                            DX_OP,
                                            DY_OP,
                                            DIntermediate_OP,
                                            UseIntermediateOut,
                                            BcastY,
                                            SameShapeOfIntermediateOutAndOut>(
          dev_ctx.stream(),
          x_data,
          y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(),
          dout->data<T>(),
          pre,
          n,
          post,
          dx_op,
          dy_op,
          dintermediate_op,
          dx == nullptr ? nullptr : dev_ctx.template Alloc<T>(dx),
          dy == nullptr ? nullptr : dev_ctx.template Alloc<T>(dy),
          dintermediate == nullptr ? nullptr
                                   : dev_ctx.template Alloc<T>(dintermediate));
#endif
    } else {
      FusedElemwiseAndActGradBroadcast2CPU<T,
                                           DX_OP,
                                           DY_OP,
                                           DIntermediate_OP,
                                           UseIntermediateOut,
                                           BcastY,
                                           SameShapeOfIntermediateOutAndOut>(
          x_data,
          y_data,
          intermediate_out == nullptr ? nullptr : intermediate_out->data<T>(),
          out->data<T>(),
          dout->data<T>(),
          pre,
          n,
          post,
          dx_op,
          dy_op,
          dintermediate_op,
          dx == nullptr ? nullptr : dev_ctx.template Alloc<T>(dx),
          dy == nullptr ? nullptr : dev_ctx.template Alloc<T>(dy),
          dintermediate == nullptr ? nullptr
                                   : dev_ctx.template Alloc<T>(dintermediate));
    }
  }
}

template <typename DeviceContext,
          typename T,
          typename DX_OP,
          typename DY_OP,
          typename DIntermediate_OP,
          bool UseIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActGradComputeEx(const DeviceContext &dev_ctx,
                                      const phi::DenseTensor *x,
                                      const phi::DenseTensor *y,
                                      const phi::DenseTensor *out,
                                      const phi::DenseTensor *intermediate_out,
                                      const phi::DenseTensor *dout,
                                      int axis,
                                      phi::DenseTensor *dx,
                                      phi::DenseTensor *dy,
                                      phi::DenseTensor *dintermediate,
                                      DX_OP dx_op,
                                      DY_OP dy_op,
                                      DIntermediate_OP dintermediate_op) {
  const phi::DDim &x_dim = x->dims();
  const phi::DDim &y_dim = y->dims();
  if (UseIntermediateOut) {
    PADDLE_ENFORCE_NOT_NULL(
        intermediate_out,
        phi::errors::InvalidArgument("Intermediate out is null pointer."));
  }
  if (x_dim == y_dim) {
    FusedElemwiseAndActGradComputeNoBroadcast<DeviceContext,
                                              T,
                                              DX_OP,
                                              DY_OP,
                                              DIntermediate_OP,
                                              UseIntermediateOut>(
        dev_ctx,
        x_dim,
        y_dim,
        x,
        y,
        intermediate_out,
        out,
        dout,
        axis,
        dx,
        dy,
        dintermediate,
        dx_op,
        dy_op,
        dintermediate_op);
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
          DeviceContext,
          T,
          DX_OP,
          DY_OP,
          DIntermediate_OP,
          UseIntermediateOut,
          true /*BcastY*/,
          SameShapeOfIntermediateOutAndOut>(dev_ctx,
                                            x_dim,
                                            y_dim,
                                            x,
                                            y,
                                            intermediate_out,
                                            out,
                                            dout,
                                            axis,
                                            dx,
                                            dy,
                                            dintermediate,
                                            dx_op,
                                            dy_op,
                                            dintermediate_op);
    } else {
      FusedElemwiseAndActGradComputeWithBroadcast<
          DeviceContext,
          T,
          DX_OP,
          DY_OP,
          DIntermediate_OP,
          UseIntermediateOut,
          false /*BcastY*/,
          SameShapeOfIntermediateOutAndOut>(dev_ctx,
                                            y_dim,
                                            x_dim,
                                            x,
                                            y,
                                            intermediate_out,
                                            out,
                                            dout,
                                            axis,
                                            dx,
                                            dy,
                                            dintermediate,
                                            dx_op,
                                            dy_op,
                                            dintermediate_op);
    }
  }
}

template <typename DeviceContext,
          typename T,
          typename CompoundFunctor,
          bool KeepIntermediateOut,
          bool SameShapeOfIntermediateOutAndOut>
void FusedElemwiseAndActComputeEx(const DeviceContext &dev_ctx,
                                  const phi::DenseTensor &x,
                                  const phi::DenseTensor &y,
                                  int axis,
                                  CompoundFunctor compound_functor,
                                  phi::DenseTensor *out,
                                  phi::DenseTensor *intermediate_out) {
  if (KeepIntermediateOut) {
    PADDLE_ENFORCE_NOT_NULL(
        intermediate_out,
        phi::errors::InvalidArgument(
            "The save_intermediate_out is opened, intermediate "
            "out is null pointer."));
  }

  const phi::DDim &x_dim = x.dims();
  const phi::DDim &y_dim = y.dims();
  if (x.dims() == y.dims()) {
    FusedElemwiseAndActComputeNoBroadcast<DeviceContext,
                                          T,
                                          CompoundFunctor,
                                          KeepIntermediateOut>(
        dev_ctx, x_dim, x, y, compound_functor, out, intermediate_out);
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
      FusedElemwiseAndActComputeWithBroadcast<DeviceContext,
                                              T,
                                              CompoundFunctor,
                                              true /*BcastY*/,
                                              KeepIntermediateOut,
                                              SameShapeOfIntermediateOutAndOut>(
          dev_ctx,
          x_dim /*OutShape*/,
          y_dim,
          x,
          y,
          compound_functor,
          axis,
          out,
          intermediate_out);
    } else {
      // In this case,
      // for 'f2(y)', the shape of intermediate_out should be equal to the
      // shape
      // of Out.
      // for 'f2(x, y)', the shape of intermediate_out should be equal to the
      // shape of Out.
      // the shape of Out should be equal to the shape of Y.
      FusedElemwiseAndActComputeWithBroadcast<DeviceContext,
                                              T,
                                              CompoundFunctor,
                                              false /*BcastY*/,
                                              KeepIntermediateOut,
                                              SameShapeOfIntermediateOutAndOut>(
          dev_ctx,
          y_dim /*OutShape*/,
          x_dim,
          x,
          y,
          compound_functor,
          axis,
          out,
          intermediate_out);
    }
  }
}

template <typename DeviceContext, typename T>
static inline void GetDoubleGradSafeTensor(const DeviceContext &dev_ctx,
                                           const phi::DenseTensor *x,
                                           const phi::DenseTensor *ddx,
                                           phi::DenseTensor *ddx_safe) {
  phi::funcs::GetDoubleGradSafeTensor<DeviceContext, T>(
      dev_ctx, *x, ddx, ddx_safe);
}

#if defined(__NVCC__) || defined(__HIPCC__)

template <typename T, typename Functor>
void GetGradXAndYOut(const phi::GPUContext &dev_ctx,
                     const phi::Place &place,
                     int axis,
                     std::vector<const phi::DenseTensor *> ins,
                     const phi::DenseTensor *dout,
                     phi::DenseTensor *dx,
                     phi::DenseTensor *dy,
                     Functor func) {
  phi::GetGradXAndYOut<T, Functor>(
      dev_ctx, place, axis, ins, *dout, dx, dy, func);
}

template <typename T, typename Functor>
void GetGradXOrYOut(const phi::GPUContext &dev_ctx,
                    const phi::Place &place,
                    int axis,
                    std::vector<const phi::DenseTensor *> ins,
                    const phi::DenseTensor *dout,
                    phi::DenseTensor *dxy,
                    Functor func) {
  phi::GetGradXOrYOut<T, Functor>(dev_ctx, place, axis, ins, *dout, dxy, func);
}

#endif

}  // namespace funcs
}  // namespace phi
