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

#include "glog/logging.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_grad_base.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T>
void ReduceWrapper(const GPUContext &dev_ctx,
                   int axis,
                   DenseTensor *src,
                   DenseTensor *dst) {
  std::vector<int> reduce_dims =
      funcs::GetReduceDim(dst->dims(), src->dims(), axis);
  phi::SumKernel<T, GPUContext>(
      dev_ctx, *src, reduce_dims, src->dtype(), false, dst);
}

template <typename T, typename Functor>
void GetGradXAndYOut(const GPUContext &dev_ctx,
                     const Place &place,
                     int axis,
                     std::vector<const DenseTensor *> ins,
                     const DenseTensor &dout,
                     DenseTensor *dx,
                     DenseTensor *dy,
                     Functor func) {
  DenseTensor tmp_dx;
  DenseTensor tmp_dy;
  dev_ctx.Alloc<T>(dx);
  dev_ctx.Alloc<T>(dy);
  std::vector<DenseTensor *> outs;
  if (dx->dims() == dout.dims() && dy->dims() == dout.dims()) {
    outs = {dx, dy};
  } else if (dx->dims() != dout.dims() && dy->dims() == dout.dims()) {
    tmp_dx.Resize(dout.dims());
    dev_ctx.Alloc<T>(&tmp_dx);
    outs = {&tmp_dx, dy};
  } else if (dx->dims() == dout.dims() && dy->dims() != dout.dims()) {
    tmp_dy.Resize(dout.dims());
    dev_ctx.Alloc<T>(&tmp_dy);
    outs = {dx, &tmp_dy};
  } else if (dx->dims() != dout.dims() && dy->dims() != dout.dims()) {
    tmp_dy.Resize(dout.dims());
    dev_ctx.Alloc<T>(&tmp_dy);
    tmp_dx.Resize(dout.dims());
    dev_ctx.Alloc<T>(&tmp_dx);
    outs = {&tmp_dx, &tmp_dy};
  }

  funcs::BroadcastKernel<T, decltype(func), 2>(dev_ctx, ins, &outs, func, axis);

  if (dx->dims() != dout.dims() && dy->dims() == dout.dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dx, dx);
  } else if (dx->dims() == dout.dims() && dy->dims() != dout.dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dy, dy);
  } else if (dx->dims() != dout.dims() && dy->dims() != dout.dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dx, dx);
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dy, dy);
  }
}

template <typename T, typename Functor>
void GetGradXOrYOut(const GPUContext &dev_ctx,
                    const Place &place,
                    int axis,
                    std::vector<const DenseTensor *> ins,
                    const DenseTensor &dout,
                    DenseTensor *dxy,
                    Functor func) {
  DenseTensor tmp_dxy;
  dev_ctx.Alloc<T>(dxy);

  std::vector<DenseTensor *> outs;
  if (dxy->dims() != dout.dims()) {
    tmp_dxy.Resize(dout.dims());
    dev_ctx.Alloc<T>(&tmp_dxy);
    outs = {&tmp_dxy};
  } else {
    outs = {dxy};
  }

  funcs::BroadcastKernel<T>(dev_ctx, ins, &outs, func, axis);
  if (dxy->dims() != dout.dims()) {
    ReduceWrapper<T>(dev_ctx, axis, &tmp_dxy, dxy);
  }
}

/*
******************************
    Add Grad
******************************
*/

template <typename T>
struct alignas(sizeof(T) * 4) Pack4 {
  T val[4];
};

template <typename T_dy, typename IndexT = int>
static __global__ void MixedPrecisionElemwiseAddGradCUDAKernel(
    const float *__restrict__ dout,
    IndexT size,
    float *__restrict__ dx,
    T_dy *__restrict__ dy) {
  IndexT tid = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
  IndexT stride = static_cast<IndexT>(gridDim.x) * blockDim.x;

  constexpr int vec_size = 4;
  IndexT loop = size / vec_size;
  IndexT remainder = size % vec_size;

  const float4 *__restrict__ dout_vec = reinterpret_cast<const float4 *>(dout);
  float4 *__restrict__ dx_vec = reinterpret_cast<float4 *>(dx);
  Pack4<T_dy> *__restrict__ dy_vec = reinterpret_cast<Pack4<T_dy> *>(dy);

  for (IndexT i = tid; i < loop; i += stride) {
    float4 val = __ldg(dout_vec + i);
    dx_vec[i] = val;

    Pack4<T_dy> dy_pack;
    dy_pack.val[0] = static_cast<T_dy>(val.x);
    dy_pack.val[1] = static_cast<T_dy>(val.y);
    dy_pack.val[2] = static_cast<T_dy>(val.z);
    dy_pack.val[3] = static_cast<T_dy>(val.w);
    dy_vec[i] = dy_pack;
  }

  if (remainder != 0) {
    IndexT tail_start = loop * vec_size;
    for (IndexT i = tail_start + tid; i < size; i += stride) {
      float val = __ldg(dout + i);
      dx[i] = val;
      dy[i] = static_cast<T_dy>(val);
    }
  }
}

template <typename T_dy>
void ElementwiseMixedPrecisionAddGrad(const GPUContext &dev_ctx,
                                      const DenseTensor &dout,
                                      DenseTensor *dx,
                                      DenseTensor *dy) {
  using T_dout = float;
  using T_dx = float;

  auto *dx_data = dev_ctx.template Alloc<T_dx>(dx);
  T_dy *dy_data = dev_ctx.template Alloc<T_dy>(dy);
  auto *dout_data = dout.data<T_dout>();

  if (dx_data == dout_data) {
    VLOG(7) << "Special case when dx_data is the same as dout_data, "
               "need cast dout to dy.";
    phi::CastKernel<T_dout>(dev_ctx, dout, dy->dtype(), dy);
    return;
  }

  auto size = dout.numel();
  if (size == 0) return;

  constexpr int vec_size = 4;
  const int64_t main_size = (size / vec_size) * vec_size;
  const int block_size = PREDEFINED_BLOCK_SIZE;
  const int grid_size =
      std::min(static_cast<int>((main_size + block_size - 1) / block_size),
               (dev_ctx.GetMaxPhysicalThreadCount() / block_size));

  dim3 grid_dim(grid_size, 1, 1);
  dim3 block_dim(block_size, 1, 1);

  if (size < std::numeric_limits<int>::max()) {
    MixedPrecisionElemwiseAddGradCUDAKernel<T_dy, int>
        <<<grid_dim, block_dim, 0, dev_ctx.stream()>>>(
            dout_data, static_cast<int>(size), dx_data, dy_data);
  } else {
    MixedPrecisionElemwiseAddGradCUDAKernel<T_dy, int64_t>
        <<<grid_dim, block_dim, 0, dev_ctx.stream()>>>(
            dout_data, static_cast<int64_t>(size), dx_data, dy_data);
  }
}

template <typename T_dy>
void DefaultMixedPrecisionAddGrad(const GPUContext &dev_ctx,
                                  const DenseTensor &x,
                                  const DenseTensor &y,
                                  const DenseTensor &dout,
                                  DenseTensor *dx,
                                  DenseTensor *dy,
                                  int axis = -1) {
  using T_dout = float;
  using T_dx = float;

  auto *dout_data = dout.data<T_dout>();

  // dx
  if (dx != nullptr) {
    auto *dx_data = dev_ctx.template Alloc<T_dx>(dx);
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
      }
    } else {
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->Resize(x.dims());
        dev_ctx.template Alloc<T_dx>(dx);
      }
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(x.dims(), dout.dims(), axis);
      phi::SumKernel<T_dout, GPUContext>(
          dev_ctx, dout, reduce_dims, dout.dtype(), false, dx);
    }
  }

  // dy
  if (dy != nullptr) {
    auto *dy_data = dev_ctx.template Alloc<T_dy>(dy);
    if (dy->dims() == dout.dims()) {
      phi::CastKernel<T_dout>(dev_ctx, dout, dy->dtype(), dy);
    } else {
      DenseTensor dy_fp32;
      dy_fp32.Resize(dout.dims());
      dev_ctx.template Alloc<float>(&dy_fp32);
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(y.dims(), dout.dims(), axis);
      phi::SumKernel<float, GPUContext>(
          dev_ctx, dout, reduce_dims, dout.dtype(), false, &dy_fp32);
      phi::CastKernel<float>(dev_ctx, dy_fp32, dy->dtype(), dy);
    }
  }
}

template <typename T, typename IndexT = int>
static __global__ void SimpleElemwiseAddGradCUDAKernel(
    const T *__restrict__ dout, IndexT size, int vec_size, T *dx, T *dy) {
  IndexT tid = static_cast<IndexT>(BLOCK_ID_X) * BLOCK_NUM_X + THREAD_ID_X;
  IndexT stride = static_cast<IndexT>(GRID_NUM_X) * BLOCK_NUM_X;
  IndexT loop = size / vec_size;
  IndexT remainder = size % vec_size;
  const float4 *dout_vec = reinterpret_cast<const float4 *>(dout);
  float4 *dx_vec = reinterpret_cast<float4 *>(dx);
  float4 *dy_vec = reinterpret_cast<float4 *>(dy);
  float4 tmp_loop;

  for (IndexT i = tid; i < loop; i += stride) {
    tmp_loop = dout_vec[i];
    dx_vec[i] = tmp_loop;
    dy_vec[i] = tmp_loop;
  }

  if (tid == loop && remainder != 0) {
    T tmp_rem;
    while (remainder) {
      IndexT idx = size - remainder;
      remainder--;
      tmp_rem = dout[idx];
      dx[idx] = tmp_rem;
      dy[idx] = tmp_rem;
    }
  }
}

template <typename T>
void DefaultElementwiseAddGrad(const GPUContext &dev_ctx,
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
    auto *dx_data = dev_ctx.template Alloc<T>(dx);
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->Resize(x.dims());
        dev_ctx.template Alloc<T>(dx);
      }
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(x.dims(), out.dims(), axis);
      phi::SumKernel<T, GPUContext>(
          dev_ctx, dout, reduce_dims, dout.dtype(), false, dx);
    }
  }
  // dy
  if (dy != nullptr) {
    auto *dy_data = dev_ctx.template Alloc<T>(dy);
    if (dy->dims() == dout.dims()) {
      if (dy_data != dout_data) {
        phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
      }
    } else {
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(y.dims(), out.dims(), axis);
      phi::SumKernel<T, GPUContext>(
          dev_ctx, dout, reduce_dims, dout.dtype(), false, dy);
    }
  }
}

template <typename T>
void ElementwiseAddGrad(const GPUContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        const DenseTensor &out,
                        const DenseTensor &dout,
                        DenseTensor *dx,
                        DenseTensor *dy) {
  dev_ctx.template Alloc<T>(dx);
  dev_ctx.template Alloc<T>(dy);
  auto *dx_data = dx->data<T>();
  auto *dy_data = dy->data<T>();
  auto *dout_data = dout.data<T>();
  if (dx_data == dout_data && dy_data != dout_data) {
    VLOG(4) << "Special case when dx_data is the same as dout_data, "
               "only need copy dout to dy";
    phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
  } else if (dx_data != dout_data && dy_data == dout_data) {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "only need copy dout to dx";
    phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
  } else if (dx_data != dout_data && dy_data != dout_data) {
    auto size = x.numel();
    int vec_size = max(static_cast<int>(sizeof(float4) / sizeof(T)), 1);
    dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
    dim3 grid_size =
        dim3(((size + vec_size - 1) / vec_size + PREDEFINED_BLOCK_SIZE - 1) /
                 PREDEFINED_BLOCK_SIZE,
             1);
    if (size < std::numeric_limits<int>::max()) {
      SimpleElemwiseAddGradCUDAKernel<T>
          <<<grid_size, block_size, 0, dev_ctx.stream()>>>(
              dout.data<T>(),
              size,
              vec_size,
              dev_ctx.template Alloc<T>(dx),
              dev_ctx.template Alloc<T>(dy));
    } else {
      SimpleElemwiseAddGradCUDAKernel<T, int64_t>
          <<<grid_size, block_size, 0, dev_ctx.stream()>>>(
              dout.data<T>(),
              size,
              vec_size,
              dev_ctx.template Alloc<T>(dx),
              dev_ctx.template Alloc<T>(dy));
    }

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
  int64_t col = static_cast<int64_t>(BLOCK_ID_X) * BLOCK_NUM_X + THREAD_ID_X;

  while (col < size) {
    if (dx != nullptr) {
      dx[col] = dout[col];
    }
    dy[col] = -dout[col];
    col += static_cast<int64_t>(BLOCK_NUM_X) * GRID_NUM_X;
  }
}

template <typename T>
void default_elementwise_sub_grad(const GPUContext &dev_ctx,
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
    auto *dx_data = dev_ctx.template Alloc<T>(dx);
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->Resize(x.dims());
        dev_ctx.template Alloc<T>(dx);
      }
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(x.dims(), out.dims(), axis);
      phi::SumKernel<T, GPUContext>(
          dev_ctx, dout, reduce_dims, dout.dtype(), false, dx);
    }
  }
  // dy
  if (dy != nullptr) {
    auto *dy_data = dev_ctx.template Alloc<T>(dy);
    if (dy->dims() == dout.dims()) {
      if (dy_data != dout_data) {
        dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
        auto size = dy->numel();
        dim3 grid_size =
            dim3((size + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
        SimpleElemwiseSubGradCUDAKernel<T>
            <<<grid_size, block_size, 0, dev_ctx.stream()>>>(
                dout.data<T>(), size, nullptr, dev_ctx.template Alloc<T>(dy));
      }
    } else {
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(y.dims(), out.dims(), axis);
      funcs::ReduceKernel<T, T, kps::AddFunctor, kps::InverseFunctor<T>>(
          dev_ctx, dout, dy, kps::InverseFunctor<T>(), reduce_dims);
    }
  }
}

template <typename T>
void elementwise_sub_grad(const GPUContext &dev_ctx,
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
  SimpleElemwiseSubGradCUDAKernel<T>
      <<<grid_size, block_size, 0, dev_ctx.stream()>>>(
          dout.data<T>(),
          size,
          dev_ctx.template Alloc<T>(dx),
          dev_ctx.template Alloc<T>(dy));
}
/*
******************************
    Div Grad
******************************
*/
template <typename T>
void ElementwiseDivGrad(const GPUContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        const DenseTensor &out,
                        const DenseTensor &dout,
                        DenseTensor *dx,
                        DenseTensor *dy,
                        int axis = -1) {
  const auto place = dev_ctx.GetPlace();
  if (dx != nullptr && dy != nullptr) {
    std::vector<const DenseTensor *> ins = {&dout, &out, &y};
    GetGradXAndYOut<T>(dev_ctx,
                       place,
                       axis,
                       ins,
                       dout,
                       dx,
                       dy,
                       funcs::DivGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    std::vector<const DenseTensor *> ins = {&dout, &y};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dx, funcs::DivGradXFunctor<T>());
  } else if (dy != nullptr && dx == nullptr) {
    std::vector<const DenseTensor *> ins = {&dout, &out, &y};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dy, funcs::DivGradYFunctor<T>());
  }
}

/*
******************************
    Mul Grad
******************************
*/

template <typename T>
void ElementwiseMulGrad(const GPUContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        const DenseTensor &dout,
                        DenseTensor *dx,
                        DenseTensor *dy,
                        int axis) {
  const auto place = dev_ctx.GetPlace();

  if (dout.numel() == 0) {
    if (dx) {
      if (dx->numel() == 0) {
        dev_ctx.template Alloc<T>(dx);
      } else {
        phi::Full<T, GPUContext>(
            dev_ctx, phi::IntArray(common::vectorize(dx->dims())), 0, dx);
      }
    }
    if (dy) {
      if (dy->numel() == 0) {
        dev_ctx.template Alloc<T>(dy);
      } else {
        phi::Full<T, GPUContext>(
            dev_ctx, phi::IntArray(common::vectorize(dy->dims())), 0, dy);
      }
    }
    return;
  }

  if (dx != nullptr && dy != nullptr) {
    std::vector<const DenseTensor *> ins = {&dout, &y, &x};
    GetGradXAndYOut<T>(dev_ctx,
                       place,
                       axis,
                       ins,
                       dout,
                       dx,
                       dy,
                       funcs::MultiplyGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    std::vector<const DenseTensor *> ins = {&dout, &y};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dx, funcs::MultiplyGradFunctor<T>());
  } else if (dx == nullptr && dy != nullptr) {
    std::vector<const DenseTensor *> ins = {&dout, &x};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, dout, dy, funcs::MultiplyGradFunctor<T>());
  }
}
}  // namespace phi
