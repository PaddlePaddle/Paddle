/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_grad_base.h"
#include "paddle/phi/kernels/gpu/reduce.h"

namespace phi {

/*
******************************
    Add Grad
******************************
*/

template <typename T>
static __global__ void SimpleElemwiseAddGradCUDAKernel(
    const T *__restrict__ dout, int size, int vec_size, T *dx, T *dy) {
  int tid = BLOCK_ID_X * BLOCK_NUM_X + THREAD_ID_X;
  int stride = GRID_NUM_X * BLOCK_NUM_X;
  int loop = size / vec_size;
  int remainder = size % vec_size;
  const float4 *dout_vec = reinterpret_cast<const float4 *>(dout);
  float4 *dx_vec = reinterpret_cast<float4 *>(dx);
  float4 *dy_vec = reinterpret_cast<float4 *>(dy);
  float4 tmp_loop;

  for (int i = tid; i < loop; i += stride) {
    tmp_loop = dout_vec[i];
    dx_vec[i] = tmp_loop;
    dy_vec[i] = tmp_loop;
  }

  if (tid == loop && remainder != 0) {
    T tmp_rem;
    while (remainder) {
      int idx = size - remainder;
      remainder--;
      tmp_rem = dout[idx];
      dx[idx] = tmp_rem;
      dy[idx] = tmp_rem;
    }
  }
}

template <typename T>
void DefaultElementwiseAddGrad(const GPUContext &ctx,
                               const DenseTensor &x,
                               const DenseTensor &y,
                               const DenseTensor &out,
                               const DenseTensor &dout,
                               DenseTensor *dx,
                               DenseTensor *dy,
                               int axis = -1) {
  auto *dout_data = dout.data<T>();

  // dx
  if (dx != nullptr) {
    auto *dx_data = dx->mutable_data<T>(ctx.GetPlace());
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        phi::Copy(ctx, dout, ctx.GetPlace(), false, dx);
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->mutable_data<T>(x.dims(), ctx.GetPlace());
      }
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(x.dims(), out.dims(), axis);
      gpuStream_t stream = ctx.stream();
      kernels::TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          ctx, dout, dx, kps::IdentityFunctor<T>(), reduce_dims, stream);
    }
  }
  // dy
  if (dy != nullptr) {
    auto *dy_data = dy->mutable_data<T>(ctx.GetPlace());
    if (dy->dims() == dout.dims()) {
      if (dy_data != dout_data) {
        phi::Copy(ctx, dout, ctx.GetPlace(), false, dy);
      }
    } else {
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(y.dims(), out.dims(), axis);
      gpuStream_t stream = ctx.stream();
      kernels::TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          ctx, dout, dy, kps::IdentityFunctor<T>(), reduce_dims, stream);
    }
  }
}

template <typename T>
void ElementwiseAddGrad(const GPUContext &ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        const DenseTensor &out,
                        const DenseTensor &dout,
                        DenseTensor *dx,
                        DenseTensor *dy) {
  ctx.template Alloc<T>(dx);
  ctx.template Alloc<T>(dy);
  auto *dx_data = dx->data<T>();
  auto *dy_data = dy->data<T>();
  auto *dout_data = dout.data<T>();
  if (dx_data == dout_data && dy_data != dout_data) {
    VLOG(4) << "Special case when dx_data is the same as dout_data, "
               "only need copy dout to dy";
    phi::Copy(ctx, dout, ctx.GetPlace(), false, dy);
  } else if (dx_data != dout_data && dy_data == dout_data) {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "only need copy dout to dx";
    phi::Copy(ctx, dout, ctx.GetPlace(), false, dx);
  } else if (dx_data != dout_data && dy_data != dout_data) {
    auto size = x.numel();
    int vec_size = max(static_cast<int>(sizeof(float4) / sizeof(T)), 1);
    dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
    dim3 grid_size =
        dim3(((size + vec_size - 1) / vec_size + PREDEFINED_BLOCK_SIZE - 1) /
                 PREDEFINED_BLOCK_SIZE,
             1);
    SimpleElemwiseAddGradCUDAKernel<
        T><<<grid_size, block_size, 0, ctx.stream()>>>(
        dout.data<T>(),
        size,
        vec_size,
        dx->mutable_data<T>(ctx.GetPlace()),
        dy->mutable_data<T>(ctx.GetPlace()));
  } else {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "and dx_data is the same as dout_data, do not need "
               "any operator";
  }
}

/*
******************************
    Sub Grad
******************************
*/

template <typename T>
static __global__ void SimpleElemwiseSubGradCUDAKernel(const T *dout,
                                                       int64_t size,
                                                       T *dx,
                                                       T *dy) {
  int col = BLOCK_ID_X * BLOCK_NUM_X + THREAD_ID_X;

  while (col < size) {
    if (dx != nullptr) {
      dx[col] = dout[col];
    }
    dy[col] = -dout[col];
    col += BLOCK_NUM_X * GRID_NUM_X;
  }
}

template <typename T>
void default_elementwise_sub_grad(const GPUContext &ctx,
                                  const DenseTensor &x,
                                  const DenseTensor &y,
                                  const DenseTensor &out,
                                  const DenseTensor &dout,
                                  DenseTensor *dx,
                                  DenseTensor *dy,
                                  int axis = -1) {
  auto *dout_data = dout.data<T>();
  // dx
  if (dx != nullptr) {
    auto *dx_data = dx->mutable_data<T>(ctx.GetPlace());
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        phi::Copy(ctx, dout, ctx.GetPlace(), false, dx);
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->mutable_data<T>(x.dims(), ctx.GetPlace());
      }
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(x.dims(), out.dims(), axis);
      gpuStream_t stream = ctx.stream();
      kernels::TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          ctx, dout, dx, kps::IdentityFunctor<T>(), reduce_dims, stream);
    }
  }
  // dy
  if (dy != nullptr) {
    auto *dy_data = dy->mutable_data<T>(ctx.GetPlace());
    if (dy->dims() == dout.dims()) {
      if (dy_data != dout_data) {
        dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
        auto size = dy->numel();
        dim3 grid_size =
            dim3((size + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
        SimpleElemwiseSubGradCUDAKernel<
            T><<<grid_size, block_size, 0, ctx.stream()>>>(
            dout.data<T>(), size, nullptr, dy->mutable_data<T>(ctx.GetPlace()));
      }
    } else {
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(y.dims(), out.dims(), axis);
      gpuStream_t stream = ctx.stream();
      kernels::TensorReduceImpl<T, T, kps::AddFunctor, kps::InverseFunctor<T>>(
          ctx, dout, dy, kps::InverseFunctor<T>(), reduce_dims, stream);
    }
  }
}

template <typename T>
void elementwise_sub_grad(const GPUContext &ctx,
                          const DenseTensor &x,
                          const DenseTensor &y,
                          const DenseTensor &out,
                          const DenseTensor &dout,
                          DenseTensor *dx,
                          DenseTensor *dy) {
  dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
  auto size = x.numel();
  dim3 grid_size =
      dim3((size + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
  SimpleElemwiseSubGradCUDAKernel<
      T><<<grid_size, block_size, 0, ctx.stream()>>>(
      dout.data<T>(),
      size,
      dx->mutable_data<T>(ctx.GetPlace()),
      dy->mutable_data<T>(ctx.GetPlace()));
}
}  // namespace phi
