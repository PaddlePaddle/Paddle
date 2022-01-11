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

// only can include the headers in paddle/pten/include dirs
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
#define GetDivMod(dividend, divisor, div, mod) \
  do {                                         \
    const auto dividend_copy = dividend;       \
    *div = dividend_copy / divisor;            \
    *mod = dividend_copy % divisor;            \
  } while (0)

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

template <typename Functor, typename T, typename OutType = T>
void CommonForwardBroadcastCPU(const framework::Tensor *x,
                               const framework::Tensor *y, framework::Tensor *z,
                               int *x_dims_array, int *y_dims_array,
                               int *out_dims_array, int max_dim,
                               const platform::CPUDeviceContext &ctx,
                               Functor func,
                               const bool is_xsize_larger = true) {
  pten::CommonForwardBroadcastCPU(x, y, z, x_dims_array, y_dims_array,
                                  out_dims_array, max_dim, ctx, func,
                                  is_xsize_larger);
}

#if defined(__NVCC__) || defined(__HIPCC__)

template <typename T, typename DX_OP, typename Tout = T>
__global__ void CommonGradBroadcastCUDAKernel(
    const int *x_strides_array, const int *y_strides_array,
    const int *out_dims_array, const int *y_strides_order,
    const int *y_dims_order, const T *x, const T *y, const Tout *out,
    const Tout *dout, T *dx, int out_size, int max_dim, int thread_num,
    DX_OP dx_op) {
  T val(0);
  int i = blockIdx.x;
  int tid = threadIdx.x;
  for (int j = tid; j < thread_num; j += blockDim.x) {
    const int X_index = i * thread_num + j;
    int out_index = X_index;
    int C_index = 0;
    int B_index = i * thread_num + j;
    int remainder = 0;
#pragma unroll
    for (int d = max_dim - 1; d >= 0; --d) {
      GetDivMod(B_index, y_dims_order[d], &B_index, &remainder);
      C_index += remainder * y_strides_order[d];
    }
    int x_index = 0;
    int y_index = 0;
    int C_index_val = C_index;
#pragma unroll
    for (int d = max_dim - 1; d >= 0; --d) {
      GetDivMod(C_index_val, out_dims_array[d], &C_index_val, &remainder);
      x_index += remainder * x_strides_array[d];
      y_index += remainder * y_strides_array[d];
    }
    out_index = C_index;
    val += dx_op(x[x_index], y[y_index], out[out_index], dout[out_index]);
  }
  val = paddle::platform::reduceSum(val, tid, thread_num);
  if (threadIdx.x == 0) {
    dx[i] = val;
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonGradBroadcastCUDA(
    const framework::Tensor &x, const framework::Tensor &y,
    const framework::Tensor &out, const framework::Tensor &dout,
    framework::Tensor *dx, framework::Tensor *dy, int *x_dims_array,
    int *y_dims_array, int *out_dims_array, int max_dim,
    const platform::CUDADeviceContext &ctx, DX_OP dx_op, DY_OP dy_op) {
  const auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
  auto cplace = platform::CPUPlace();
  const T *x_data = x.data<T>();
  const T *y_data = y.data<T>();
  const Tout *out_data = out.data<Tout>();
  const Tout *dout_data = dout.data<Tout>();
  T *dx_data = dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace());
  T *dy_data = dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace());

  std::vector<int> x_one_indexs;
  std::vector<int> y_one_indexs;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] != y_dims_array[i]) {
      if (x_dims_array[i] == 1) {
        x_one_indexs.push_back(i);
      }
      if (y_dims_array[i] == 1) {
        y_one_indexs.push_back(i);
      }
    }
  }

  std::vector<int> x_trans_indexs(max_dim);
  std::vector<int> y_trans_indexs(max_dim);
  pten::ComputeBroadcastTranspositionArray(
      x_one_indexs.data(), x_trans_indexs.data(), max_dim, x_one_indexs.size());
  pten::ComputeBroadcastTranspositionArray(
      y_one_indexs.data(), y_trans_indexs.data(), max_dim, y_one_indexs.size());

  // compute array stride for cuda kernel;
  // e.g. x.dims=[2,3,4], x_stride=[12,4,1]
  std::vector<int> x_strides_array(max_dim);
  std::vector<int> y_strides_array(max_dim);
  std::vector<int> out_strides_array(max_dim);
  int x_stride = 1;
  int y_stride = 1;
  int z_stride = 1;
  for (int i = max_dim - 1; i >= 0; i--) {
    x_strides_array[i] = x_dims_array[i] == 1 ? 0 : x_stride;
    y_strides_array[i] = y_dims_array[i] == 1 ? 0 : y_stride;
    out_strides_array[i] = z_stride;
    x_stride *= x_dims_array[i];
    y_stride *= y_dims_array[i];
    z_stride *= out_dims_array[i];
  }

  std::vector<int> x_strides_order(max_dim);
  std::vector<int> y_strides_order(max_dim);
  std::vector<int> x_dims_order(max_dim);
  std::vector<int> y_dims_order(max_dim);
  for (int i = 0; i < max_dim; ++i) {
    x_strides_order[i] = out_strides_array[x_trans_indexs[i]];
    y_strides_order[i] = out_strides_array[y_trans_indexs[i]];
    x_dims_order[i] = out_dims_array[x_trans_indexs[i]];
    y_dims_order[i] = out_dims_array[y_trans_indexs[i]];
  }
  std::vector<int> x_broadcast_pos;
  std::vector<int> y_broadcast_pos;

  int bytes = max_dim * sizeof(int);

  for (int i = 0; i < max_dim; ++i) {
    if (x_dims_array[i] != out_dims_array[i] && x_dims_array[i] == 1) {
      x_broadcast_pos.emplace_back(i);
    }
    if (y_dims_array[i] != out_dims_array[i] && y_dims_array[i] == 1) {
      y_broadcast_pos.emplace_back(i);
    }
  }

  auto stream = ctx.stream();
  bool can_split_x = false;
  bool can_split_y = false;

  auto FastCommonCUDAF = [&](const std::vector<int> &broadcast_pos, bool is_y) {
    int h =
        std::accumulate(out_dims_array, out_dims_array + broadcast_pos.size(),
                        1, std::multiplies<int>());
    int w =
        std::accumulate(out_dims_array + broadcast_pos.size(),
                        out_dims_array + max_dim, 1, std::multiplies<int>());

    VLOG(3) << "FastCommonCUDAF elementwise w:" << w << " h:" << h
            << " is_y:" << is_y;

    int split_h;
    int split_w;
    int kh = h;
    int kw = w;

    if (is_y) {
      split_h =
          std::accumulate(x_dims_array, x_dims_array + broadcast_pos.size(), 1,
                          std::multiplies<int>());
      split_w =
          std::accumulate(x_dims_array + broadcast_pos.size(),
                          x_dims_array + max_dim, 1, std::multiplies<int>());

    } else {
      split_h =
          std::accumulate(y_dims_array, y_dims_array + broadcast_pos.size(), 1,
                          std::multiplies<int>());
      split_w =
          std::accumulate(y_dims_array + broadcast_pos.size(),
                          y_dims_array + max_dim, 1, std::multiplies<int>());
    }

    if (h > split_h) kh = split_h;
    if (w > split_w) kw = split_w;

    if (is_y) {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        pten::CommonGradBroadcast1CUDAKernelHeight<<<grid_size, block_size, 0,
                                                     stream>>>(
            x_data, y_data, out_data, dout_data, h, w, dy_op, dy_data, kh, kw,
            is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
        pten::FastCommonGradBroadcastCUDAKernelHeight<<<grid_size, block_size,
                                                        0, stream>>>(
            x_data, y_data, out_data, dout_data, h, w, dy_op, dy_data, kh, kw,
            is_y);
      }
    } else {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        pten::CommonGradBroadcast1CUDAKernelHeight<<<grid_size, block_size, 0,
                                                     stream>>>(
            x_data, y_data, out_data, dout_data, h, w, dx_op, dx_data, kh, kw,
            is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
        pten::FastCommonGradBroadcastCUDAKernelHeight<<<grid_size, block_size,
                                                        0, stream>>>(
            x_data, y_data, out_data, dout_data, h, w, dx_op, dx_data, kh, kw,
            is_y);
      }
    }
  };

  auto FastBroadCastHeightCUDAF = [&](const std::vector<int> &broadcast_pos,
                                      bool x_large) {
    int h =
        std::accumulate(out_dims_array, out_dims_array + broadcast_pos.size(),
                        1, std::multiplies<int>());
    int w =
        std::accumulate(out_dims_array + broadcast_pos.size(),
                        out_dims_array + max_dim, 1, std::multiplies<int>());

    VLOG(3) << "FastBroadCastHeightCUDAF w:" << w << " h:" << h;

    if (w < 16 || h < 16) {
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
      int grid_size = w;
      pten::ElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0,
                                               stream>>>(
          x_data, y_data, out_data, dout_data, h, w, x_large, dx_op, dy_op,
          dx_data, dy_data);
    } else {
      dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
      int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
      pten::FastElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0,
                                                   stream>>>(
          x_data, y_data, out_data, dout_data, h, w, x_large, dx_op, dy_op,
          dx_data, dy_data);
    }
  };

  auto FastBroadCastAllCUDAF = [&](const std::vector<int> &broadcast_pos,
                                   int max_dim, bool is_x_large) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(out_dims_array, out_dims_array + axis, 1,
                              std::multiplies<int>());
    int mid = 1;
    int post = 1;

    if (broadcast_pos.size() == 1) {
      mid = out_dims_array[axis];
      post =
          std::accumulate(out_dims_array + axis + 1, out_dims_array + max_dim,
                          1, std::multiplies<int>());
    } else {
      mid = std::accumulate(out_dims_array + axis,
                            out_dims_array + broadcast_pos.back() + 1, 1,
                            std::multiplies<int>());
      post =
          std::accumulate(out_dims_array + broadcast_pos.back() + 1,
                          out_dims_array + max_dim, 1, std::multiplies<int>());
    }

    VLOG(3) << "FastBroadCastAllCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;

    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
    int grid_size = pre * post;

    pten::FastCommonGradBroadcastAllCUDAKernel<<<grid_size, block_size, 0,
                                                 stream>>>(
        x_data, y_data, out_data, dout_data, pre, mid, post, is_x_large, dx_op,
        dy_op, dx_data, dy_data);
  };

  auto FastBroadCastOneCUDAF = [&](const std::vector<int> &broadcast_pos,
                                   int max_dim, bool is_x) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(out_dims_array, out_dims_array + axis, 1,
                              std::multiplies<int>());
    int mid = out_dims_array[axis];
    int post =
        std::accumulate(out_dims_array + axis + 1, out_dims_array + max_dim, 1,
                        std::multiplies<int>());

    int k_pre;
    int k_mid;
    int k_post;

    if (is_x) {
      k_pre = std::accumulate(y_dims_array, y_dims_array + axis, 1,
                              std::multiplies<int>());
      k_mid = y_dims_array[axis];
      k_post = std::accumulate(y_dims_array + axis + 1, y_dims_array + max_dim,
                               1, std::multiplies<int>());
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
      int grid_size = pre * post;
      // we need to calc y offset with blockid, so do x_pre/y_pre to get left
      // size.
      if (k_pre != pre) k_pre = pre / k_pre;

      pten::FastCommonGradBroadcastOneCUDAKernel<<<grid_size, block_size, 0,
                                                   stream>>>(
          x_data, y_data, out_data, dout_data, pre, mid, post, k_pre, k_mid,
          k_post, true, dx_op, dx_data);
    } else {
      k_pre = std::accumulate(x_dims_array, x_dims_array + axis, 1,
                              std::multiplies<int>());
      k_mid = x_dims_array[axis];
      k_post = std::accumulate(x_dims_array + axis + 1, x_dims_array + max_dim,
                               1, std::multiplies<int>());
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
      int grid_size = pre * post;
      if (k_pre != pre) k_pre = pre / k_pre;

      pten::FastCommonGradBroadcastOneCUDAKernel<<<grid_size, block_size, 0,
                                                   stream>>>(
          x_data, y_data, out_data, dout_data, pre, mid, post, k_pre, k_mid,
          k_post, false, dy_op, dy_data);
    }
    VLOG(3) << "FastBroadCastOneCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;
  };

  // do fast elementwise if: 1. only one input need to do broadcast, we can
  // fallback
  // to old fast path.
  // 2. if both x and y need broadcast, then do it one by one.
  bool fast_broadcast = false;
  if (x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    can_split_y = pten::SplitDims(y_broadcast_pos, max_dim);
    if (can_split_y) {
      // only y need to do broadcast on h
      if (y_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(y_broadcast_pos, true);
        fast_broadcast = true;
      }
    } else if (y_broadcast_pos.size() == 1 ||
               pten::CheckContiguousDims(
                   y_broadcast_pos)) {  // for only one dim and
                                        // contiguous broadcast.
      // If cannot split,  which means input has 3 parts
      FastBroadCastAllCUDAF(y_broadcast_pos, max_dim, true);
      fast_broadcast = true;
    }
  } else if (y_broadcast_pos.empty() && !x_broadcast_pos.empty()) {
    // only x need broadcast
    can_split_x = pten::SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(x_broadcast_pos, false);
        fast_broadcast = true;
      }
    } else if (x_broadcast_pos.size() == 1 ||
               pten::CheckContiguousDims(x_broadcast_pos)) {
      FastBroadCastAllCUDAF(x_broadcast_pos, max_dim, false);
      fast_broadcast = true;
    }
  } else if (!x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    // do x and y broadcast each.
    can_split_y = pten::SplitDims(y_broadcast_pos, max_dim);
    bool fast_broadcast_x = false;
    bool fast_broadcast_y = false;
    if (can_split_y) {
      // begin at start.
      if (y_broadcast_pos[0] == 0) {
        FastCommonCUDAF(y_broadcast_pos, true);
        fast_broadcast_y = true;
      }
    } else if (y_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(y_broadcast_pos, max_dim, false);
      can_split_y = true;
      fast_broadcast_y = true;
    }
    can_split_x = pten::SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastCommonCUDAF(x_broadcast_pos, false);
        fast_broadcast_x = true;
      }
    } else if (x_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(x_broadcast_pos, max_dim, true);
      can_split_x = true;
      fast_broadcast_x = true;
    }
    VLOG(3) << "CommonBroadcast can_split_y:" << can_split_y
            << " can_split_x:" << can_split_x;
    // if both x and y into fast path then return
    if (fast_broadcast_x && fast_broadcast_y) {
      fast_broadcast = true;
    }
    if (can_split_y && can_split_x && fast_broadcast) return;
  }

  // Should remove memory copy, use reg instead.
  if (fast_broadcast) {
    return;
  }
  int x_blocks = 0;
  int x_threads = 0;
  pten::ComputeBroadcastKernelSize(x_dims_array, out_dims_array, &x_blocks,
                                   &x_threads, max_dim);
  int y_blocks = 0;
  int y_threads = 0;
  pten::ComputeBroadcastKernelSize(y_dims_array, out_dims_array, &y_blocks,
                                   &y_threads, max_dim);

  auto x_strides_array_tmp = memory::Alloc(ctx, bytes);
  int *x_strides_array_gpu =
      reinterpret_cast<int *>(x_strides_array_tmp->ptr());
  memory::Copy(gplace, x_strides_array_gpu, cplace, x_strides_array.data(),
               bytes, ctx.stream());

  auto y_strides_array_tmp = memory::Alloc(ctx, bytes);
  int *y_strides_array_gpu =
      reinterpret_cast<int *>(y_strides_array_tmp->ptr());
  memory::Copy(gplace, y_strides_array_gpu, cplace, y_strides_array.data(),
               bytes, ctx.stream());

  auto out_dims_array_tmp = memory::Alloc(ctx, bytes);
  int *out_dims_array_gpu = reinterpret_cast<int *>(out_dims_array_tmp->ptr());
  memory::Copy(gplace, out_dims_array_gpu, cplace, out_dims_array, bytes,
               ctx.stream());

  const int out_size = std::accumulate(out_dims_array, out_dims_array + max_dim,
                                       1, std::multiplies<int>());
  int x_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, x_threads);
  int y_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, y_threads);
  if (dx) {
    auto x_strides_order_tmp = memory::Alloc(ctx, bytes);
    int *x_strides_order_gpu =
        reinterpret_cast<int *>(x_strides_order_tmp->ptr());
    memory::Copy(gplace, x_strides_order_gpu, cplace, x_strides_order.data(),
                 bytes, ctx.stream());

    auto x_dims_order_tmp = memory::Alloc(ctx, bytes);
    int *x_dims_order_gpu = reinterpret_cast<int *>(x_dims_order_tmp->ptr());
    memory::Copy(gplace, x_dims_order_gpu, cplace, x_dims_order.data(), bytes,
                 ctx.stream());
    CommonGradBroadcastCUDAKernel<
        T, DX_OP, Tout><<<x_blocks, x_block_size, 0, ctx.stream()>>>(
        x_strides_array_gpu, y_strides_array_gpu, out_dims_array_gpu,
        x_strides_order_gpu, x_dims_order_gpu, x_data, y_data, out_data,
        dout_data, dx_data, out_size, max_dim, x_threads, dx_op);
  }
  if (dy) {
    auto y_strides_order_tmp = memory::Alloc(ctx, bytes);
    int *y_strides_order_gpu =
        reinterpret_cast<int *>(y_strides_order_tmp->ptr());
    memory::Copy(gplace, y_strides_order_gpu, cplace, y_strides_order.data(),
                 bytes, ctx.stream());

    auto y_dims_order_tmp = memory::Alloc(ctx, bytes);
    int *y_dims_order_gpu = reinterpret_cast<int *>(y_dims_order_tmp->ptr());
    memory::Copy(gplace, y_dims_order_gpu, cplace, y_dims_order.data(), bytes,
                 ctx.stream());
    CommonGradBroadcastCUDAKernel<
        T, DY_OP, Tout><<<y_blocks, y_block_size, 0, ctx.stream()>>>(
        x_strides_array_gpu, y_strides_array_gpu, out_dims_array_gpu,
        y_strides_order_gpu, y_dims_order_gpu, x_data, y_data, out_data,
        dout_data, dy_data, out_size, max_dim, y_threads, dy_op);
  }
}

#endif  // __NVCC__ or __HIPCC__

inline framework::DDim trim_trailing_singular_dims(
    const framework::DDim &dims) {
  return pten::funcs::trim_trailing_singular_dims(dims);
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          typename Tout = T>
void CommonElementwiseBroadcastBackward(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dims,
    const framework::DDim &y_dims, const framework::Tensor &x,
    const framework::Tensor &y, const framework::Tensor &out,
    const framework::Tensor &dout, int axis, framework::Tensor *dx,
    framework::Tensor *dy, DX_OP dx_op, DY_OP dy_op) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                         y_dims_array.data(), out_dims_array.data(), max_dim,
                         axis);
  // for inplace strategy. memset will make dx and dout clear and get wrong
  // result.
  if (dx && dx->IsSharedBufferWith(dout)) {
    dx->clear();
    dx->mutable_data<T>(x_dims, ctx.GetPlace());
  }

  VLOG(3) << "CommonElementwiseBroadcastBackward xdims:"
          << framework::make_ddim(x_dims_array)
          << " ydim:" << framework::make_ddim(y_dims_array);

  if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
    CommonGradBroadcastCUDA<T, DX_OP, DY_OP, Tout>(
        x, y, out, dout, dx, dy, x_dims_array.data(), y_dims_array.data(),
        out_dims_array.data(), max_dim,
        ctx.template device_context<platform::CUDADeviceContext>(), dx_op,
        dy_op);
#endif
  } else {
    pten::CommonGradBroadcastCPU<T, DX_OP, DY_OP, Tout>(
        x, y, out, dout, dx, dy, x_dims_array.data(), y_dims_array.data(),
        out_dims_array.data(), max_dim,
        ctx.template device_context<platform::CPUDeviceContext>(), dx_op,
        dy_op);
  }
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP,
          typename Tout = T>
void ElemwiseGradComputeWithBroadcast(
    const framework::ExecutionContext &ctx, const framework::DDim &x_dims,
    const framework::DDim &y_dims, const framework::Tensor &x,
    const framework::Tensor &y, const framework::Tensor &out,
    const framework::Tensor &dout, int axis, framework::Tensor *dx,
    framework::Tensor *dy, DX_OP dx_op, DY_OP dy_op) {
  bool is_xsize_larger = true;

  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis, 0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis, max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim, axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimed = trim_trailing_singular_dims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    pten::funcs::get_mid_dims(x_dims, y_dims_trimed, axis_trim, &pre, &n, &post,
                              &is_run_common_broadcast);
  } else {
    auto x_dims_trimed = trim_trailing_singular_dims(x_dims);
    axis_trim = (x_dims_trimed.size() == 0) ? y_dims.size() : axis;
    pten::funcs::get_mid_dims(y_dims, x_dims_trimed, axis_trim, &pre, &n, &post,
                              &is_run_common_broadcast);
  }
  // special case for common backward implementation.
  if (is_run_common_broadcast) {
    CommonElementwiseBroadcastBackward<DeviceContext, T, DX_OP, DY_OP, Tout>(
        ctx, x_dims, y_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
    return;
  }
  if (post == 1) {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      pten::ElemwiseGradBroadcast1CUDA(
          ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
          y.data<T>(), out.data<Tout>(), dout.data<Tout>(), pre, n,
          is_xsize_larger, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      pten::ElemwiseGradBroadcast1CPU(
          x.data<T>(), y.data<T>(), out.data<Tout>(), dout.data<Tout>(), pre, n,
          is_xsize_larger, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
    }
  } else {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      pten::ElemwiseGradBroadcast2CUDA(
          ctx.template device_context<DeviceContext>().stream(), x.data<T>(),
          y.data<T>(), out.data<Tout>(), dout.data<Tout>(), pre, n, post,
          is_xsize_larger, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
#endif
    } else {
      pten::ElemwiseGradBroadcast2CPU(
          x.data<T>(), y.data<T>(), out.data<Tout>(), dout.data<Tout>(), pre, n,
          post, is_xsize_larger, dx_op, dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(ctx.GetPlace()),
          dy == nullptr ? nullptr : dy->mutable_data<T>(ctx.GetPlace()));
    }
  }
}

template <typename Functor, typename DeviceContext, typename T,
          typename OutType = T>
void CommonElementwiseBroadcastForward(
    const framework::ExecutionContext &ctx, const framework::Tensor *x,
    const framework::Tensor *y, framework::Tensor *z,
    const framework::DDim &x_dims, const framework::DDim &y_dims, Functor func,
    int axis, const bool is_xsize_larger = true) {
  z->mutable_data<OutType>(ctx.GetPlace());
  auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
  auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);
  auto pt_z = paddle::experimental::MakePtenDenseTensor(*z);
  const auto &dev_ctx = ctx.template device_context<DeviceContext>();
  pten::CommonElementwiseBroadcastForward(dev_ctx, *pt_x.get(), *pt_y.get(),
                                          pt_z.get(), x_dims, y_dims, func,
                                          axis, is_xsize_larger);
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
  if (x.dims() == y.dims()) {
    const auto &dev_ctx = ctx.template device_context<DeviceContext>();
    pten::funcs::ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP,
                                                Tout>(
        dev_ctx, x_dim, y_dim, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
  } else {
    ElemwiseGradComputeWithBroadcast<DeviceContext, T, DX_OP, DY_OP, Tout>(
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
  const framework::DDim &x_dim = x.dims();
  const framework::DDim &y_dim = y.dims();
  if (x.dims() == y.dims()) {
    const auto &dev_ctx = ctx.template device_context<DeviceContext>();
    pten::funcs::ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        dev_ctx, x_dim, y_dim, dout, dout, out, dout, axis, dx, dy, dx_op,
        dy_op);
  } else {
    ElemwiseGradComputeWithBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dim, y_dim, dout, dout, out, dout, axis, dx, dy, dx_op, dy_op);
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
  z->mutable_data<OutType>(ctx.GetPlace());
  auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
  auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);
  auto pt_z = paddle::experimental::MakePtenDenseTensor(*z);
  if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
    std::vector<const framework::Tensor *> ins = {x, y};
    std::vector<framework::Tensor *> outs = {z};
    z->mutable_data<OutType>(ctx.GetPlace());

    const auto &dev_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, OutType>(
        dev_ctx, ins, &outs, axis, func);
#endif
    return;
  }

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

  LaunchElementwiseCudaKernel<ET, T, T, decltype(func), 2>(dev_ctx, ins, &outs,
                                                           axis, func);

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

  LaunchElementwiseCudaKernel<ET, T, T>(dev_ctx, ins, &outs, axis, func);
  if (dxy->dims() != dout->dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dxy, dxy);
  }
}

#endif

}  // namespace operators
}  // namespace paddle
