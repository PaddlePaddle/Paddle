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
#include <utility>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
void default_elementwise_add(const framework::ExecutionContext &ctx,
                             const framework::Tensor *x,
                             const framework::Tensor *y, framework::Tensor *z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          AddFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseAddFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseAddFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T, class Enable = void>
struct SameDimsElemwiseAdd {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z);
};

template <typename DeviceContext, typename T>
class ElementwiseAddKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    auto *y = ctx.Input<framework::LoDTensor>("Y");
    auto *z = ctx.Output<framework::LoDTensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());
    auto dims_equal = x->dims() == y->dims();
    if (dims_equal) {
      SameDimsElemwiseAdd<DeviceContext, T> same_dims_add;
      same_dims_add(ctx, x, y, z);
    } else {
      default_elementwise_add<DeviceContext, T>(ctx, x, y, z);
    }
  }
};

template <typename T>
struct IdentityGrad {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename DeviceContext, typename T>
void default_elementwise_add_grad(const framework::ExecutionContext &ctx,
                                  const framework::Tensor *x,
                                  const framework::Tensor *y,
                                  const framework::Tensor *out,
                                  const framework::Tensor *dout,
                                  framework::Tensor *dx,
                                  framework::Tensor *dy) {
  int axis = ctx.Attr<int>("axis");

  ElemwiseExplicitGradCompute<DeviceContext, T, IdentityGrad<T>,
                              IdentityGrad<T>>(ctx, *x, *y, *out, *dout, axis,
                                               dx, dy, IdentityGrad<T>(),
                                               IdentityGrad<T>());
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_add_grad(const framework::ExecutionContext &ctx,
                     const framework::Tensor *x, const framework::Tensor *y,
                     const framework::Tensor *out,
                     const framework::Tensor *dout, framework::Tensor *dx,
                     framework::Tensor *dy) {
  auto blas = math::GetBlas<DeviceContext, T>(ctx);
  if (dx) {
    blas.VCOPY(dout->numel(), dout->data<T>(),
               dx->mutable_data<T>(ctx.GetPlace()));
  }

  if (dy) {
    blas.VCOPY(dout->numel(), dout->data<T>(),
               dy->mutable_data<T>(ctx.GetPlace()));
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    !std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_add_grad(const framework::ExecutionContext &ctx,
                     const framework::Tensor *x, const framework::Tensor *y,
                     const framework::Tensor *out,
                     const framework::Tensor *dout, framework::Tensor *dx,
                     framework::Tensor *dy) {
  default_elementwise_add_grad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
}

#ifdef PADDLE_WITH_CUDA
#ifdef __NVCC__

template <typename T, int BLOCK_W, int BLOCK_H>
__global__ void MatrixColReduce(const T *__restrict__ in, T *__restrict__ out,
                                size_t width, size_t height) {
  __shared__ T sdata[BLOCK_H][BLOCK_W + 1];
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  size_t width_stride = gridDim.x * blockDim.x;
  size_t full_width = (width & (~((uint64_t)(BLOCK_W - 1)))) +
                      ((width & (BLOCK_W - 1)) ? BLOCK_W : 0);

#pragma unroll
  for (size_t w = idx; w < full_width; w += width_stride) {
    sdata[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();
    size_t offset = w + threadIdx.y * width;
#pragma unroll
    for (size_t h = threadIdx.y; h < height;
         h += BLOCK_H) {  // block-stride loop across matrix height
      sdata[threadIdx.y][threadIdx.x] +=
          (w < width) ? in[offset] : (static_cast<T>(0));
      offset += width * BLOCK_H;
    }
    __syncthreads();

    T val = sdata[threadIdx.x][threadIdx.y];
    for (int i = warpSize >> 1; i > 0; i >>= 1)
      val += platform::CudaShuffleXorSync(0xFFFFFFFF, val, i);

    __syncthreads();
    if (threadIdx.x == 0) sdata[0][threadIdx.y] = val;
    __syncthreads();
    if ((threadIdx.y == 0) && ((w) < width)) out[w] = sdata[0][threadIdx.x];
  }
}

template <int BLOCK_W, int BLOCK_H>
__global__ void FP16MatrixColReduce(
    const paddle::platform::float16 *__restrict__ in,
    paddle::platform::float16 *__restrict__ out, size_t width, size_t height) {
  constexpr int repeats = BLOCK_H / BLOCK_W;
  __shared__ paddle::platform::float16 sdata[BLOCK_H][BLOCK_W + 1];
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  size_t width_stride = gridDim.x * blockDim.x;
  size_t full_width = (width & (~((uint64_t)(BLOCK_W - 1)))) +
                      ((width & (BLOCK_W - 1)) ? BLOCK_W : 0);

#pragma unroll
  for (size_t w = idx; w < full_width; w += width_stride) {
    for (int r = 0; r < repeats; r++) {
      sdata[threadIdx.y + r * BLOCK_W][threadIdx.x] = 0;
    }
    __syncthreads();
    for (int r = 0; r < repeats; r++) {
      size_t offset = w + (r * BLOCK_W + threadIdx.y) * width;
#pragma unroll
      for (size_t h = r * BLOCK_H + threadIdx.y; h < height;
           h += BLOCK_H) {  // block-stride loop across matrix height
        sdata[r * BLOCK_W + threadIdx.y][threadIdx.x] +=
            (w < width) ? in[offset + r * BLOCK_W * width]
                        : (static_cast<paddle::platform::float16>(0));
        offset += width * BLOCK_H;
      }
    }
    __syncthreads();

    paddle::platform::float16 result =
        static_cast<paddle::platform::float16>(0);
    for (int r = 0; r < repeats; r++) {
      paddle::platform::float16 val =
          sdata[threadIdx.x + r * BLOCK_W][threadIdx.y];
      for (int i = warpSize >> 1; i > 0; i >>= 1)
        val += platform::CudaShuffleXorSync(0xFFFFFFFF, val, i);
      __syncthreads();
      result += val;
    }
    if (threadIdx.x == 0) sdata[0][threadIdx.y] = result;
    __syncthreads();
    if ((threadIdx.y == 0) && ((w) < width)) out[w] = sdata[0][threadIdx.x];
  }
}
#endif
#endif
bool static RunSpecialDims(const framework::DDim &dx_dims,
                           const framework::DDim &dy_dims,
                           const framework::DDim &dout_dims, int axis) {
  auto smaller_dims = dx_dims;
  auto bigger_dims = dy_dims;
  auto smaller_dims_size = smaller_dims.size();
  auto bigger_dims_size = bigger_dims.size();
  int smaller_ignore_size = 0;
  int bigger_ignore_size = 0;
  for (int i = 0; i < smaller_dims_size; i++) {
    if (smaller_dims[i] == 1)
      smaller_ignore_size++;
    else
      break;
  }
  for (int i = 0; i < bigger_dims_size; i++) {
    if (bigger_dims[i] == 1)
      bigger_ignore_size++;
    else
      break;
  }

  int smaller_real_size = smaller_dims.size() - smaller_ignore_size;
  int bigger_real_size = bigger_dims.size() - bigger_ignore_size;

  if (smaller_real_size == bigger_real_size) return false;

  if (bigger_real_size < smaller_real_size) {
    smaller_dims = dy_dims;
    bigger_dims = dx_dims;
    std::swap(smaller_real_size, bigger_real_size);
  }
  int big_size = bigger_dims.size();
  int small_size = smaller_dims.size();
  for (int i = 1; i <= smaller_real_size; i++) {
    if (bigger_dims[big_size - i] != smaller_dims[small_size - i]) return false;
  }

  if (axis != -1 && (axis != (bigger_real_size - smaller_real_size))) {
    return false;
  }

  return true;
}

#ifdef PADDLE_WITH_CUDA
// cuda definition
template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
elementwise_add_grad(const framework::ExecutionContext &ctx,
                     const framework::Tensor *x, const framework::Tensor *y,
                     const framework::Tensor *out,
                     const framework::Tensor *dout, framework::Tensor *dx,
                     framework::Tensor *dy);
#endif

template <typename DeviceContext, typename T>
class ElementwiseAddGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);

    using Tensor = framework::Tensor;

    auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Input<Tensor>("Y");
    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    // skip out
    auto *out = dout;

#ifdef PADDLE_WITH_CUDA
#ifdef __NVCC__

    int axis = ctx.Attr<int>("axis");
    if (ctx.GetPlace() == platform::CUDAPlace() && dx != nullptr &&
        dy != nullptr && dout != nullptr && dx->numel() != dy->numel() &&
        RunSpecialDims(dx->dims(), dy->dims(), dout->dims(), axis)) {
      auto *dx_data = dx->mutable_data<T>(ctx.GetPlace());
      auto *dy_data = dy->mutable_data<T>(ctx.GetPlace());
      auto *dout_data = dout->data<T>();
      auto stream = ctx.cuda_device_context().stream();
      auto *out_data = dx_data;
      int width = dx->numel();
      int height = dout->numel() / width;
      if (dx->dims() == dout->dims()) {
        width = dy->numel();
        height = dout->numel() / width;
        out_data = dy_data;
        framework::TensorCopy(
            *dout, ctx.GetPlace(),
            ctx.template device_context<platform::DeviceContext>(), dx);
      } else {
        framework::TensorCopy(
            *dout, ctx.GetPlace(),
            ctx.template device_context<platform::DeviceContext>(), dy);
      }

      constexpr int block_x = 32;
      constexpr int block_y = 32;
      dim3 blocks(block_x, block_y);

      int max_physical_threads =
          ctx.cuda_device_context().GetMaxPhysicalThreadCount();
      int max_blocks = std::max(max_physical_threads / (block_x * block_y), 1);
      int theory_block = (width + blocks.x - 1) / blocks.x;
      dim3 grids(std::min(theory_block, max_blocks));
      if (std::is_same<T, paddle::platform::float16>::value) {
        const paddle::platform::float16 *ptr1 =
            reinterpret_cast<const paddle::platform::float16 *>(dout_data);
        paddle::platform::float16 *ptr2 =
            reinterpret_cast<paddle::platform::float16 *>(out_data);
        if (height <= 32) {
          FP16MatrixColReduce<32, 32><<<grids, blocks, 0, stream>>>(
              ptr1, ptr2, width, height);
        } else {
          FP16MatrixColReduce<32, 64><<<grids, blocks, 0, stream>>>(
              ptr1, ptr2, width, height);
        }
        return;
      }
      MatrixColReduce<T, block_x, block_y><<<grids, blocks, 0, stream>>>(
          dout_data, out_data, width, height);
      return;
    }

#endif
#endif
    // Special case when dy is not needed and dx doesn't reduce
    if (dx != nullptr && dy == nullptr && dx->dims() == dout->dims()) {
      VLOG(4) << "Special case when dy is not needed and dx doesn't "
                 "reduce";
      framework::TensorCopy(
          *dout, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), dx);
    } else if (dx == nullptr && dy != nullptr && dy->dims() == dout->dims()) {
      VLOG(4) << "Special case when dx is not needed and dy doesn't "
                 "reduce";
      framework::TensorCopy(
          *dout, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), dy);
    } else if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
      elementwise_add_grad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
    } else {
      default_elementwise_add_grad<DeviceContext, T>(ctx, x, y, out, dout, dx,
                                                     dy);
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseAddDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using Tensor = framework::Tensor;

    auto *y = ctx.Input<Tensor>("Y");
    auto *dout = ctx.Input<Tensor>("DOut");
    auto *ddx = ctx.Input<Tensor>("DDX");
    auto *ddy = ctx.Input<Tensor>("DDY");

    auto *ddout = ctx.Output<Tensor>("DDOut");

    // ddOut = ddx + ddy
    if (ddout) {
      Tensor ddx_safe, ddy_safe;
      GetDoubleGradSafeTensor<DeviceContext, T>(ctx, dout, ddx, &ddx_safe);
      GetDoubleGradSafeTensor<DeviceContext, T>(ctx, y, ddy, &ddy_safe);

      ddout->mutable_data<T>(ctx.GetPlace());
      default_elementwise_add<DeviceContext, T>(ctx, &ddx_safe, &ddy_safe,
                                                ddout);
    }
  }
};

}  // namespace operators
}  // namespace paddle
