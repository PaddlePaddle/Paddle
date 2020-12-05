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
#include <algorithm>
#include <functional>
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

#define WARPSIZE 32

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
struct SameDimsElemwiseAdd<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    AddRangeFunctor<T> functor(x->data<T>(), y->data<T>(), z->data<T>());
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx,
                                                              x->numel());
    for_range(functor);
  }
};

template <>
struct SameDimsElemwiseAdd<platform::CUDADeviceContext, platform::float16> {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z) {
    auto size = x->numel();
    dim3 grid_size = dim3(((size + 1) / 2 + PADDLE_CUDA_THREAD_SIZE - 1) /
                              PADDLE_CUDA_THREAD_SIZE,
                          1);
    dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
    const half* x2 =
        reinterpret_cast<const half*>(x->data<platform::float16>());
    const half* y2 =
        reinterpret_cast<const half*>(y->data<platform::float16>());
    half* z2 = reinterpret_cast<half*>(z->data<platform::float16>());
    SameDimsElemwiseAddCUDAKernel<<<
        grid_size, block_size, 0,
        ctx.template device_context<platform::CUDADeviceContext>().stream()>>>(
        x2, y2, z2, size);
  }
};

template <typename T>
static __global__ void SimpleElemwiseAddGradCUDAKernel(const T* dout,
                                                       int64_t size, T* dx,
                                                       T* dy) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  while (col < size) {
    dx[col] = dout[col];
    dy[col] = dout[col];
    col += blockDim.x * gridDim.x;
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, plat::CUDADeviceContext>::value>::type
ElementwiseAddGrad(const framework::ExecutionContext& ctx,
                   const framework::Tensor* x, const framework::Tensor* y,
                   const framework::Tensor* out, const framework::Tensor* dout,
                   framework::Tensor* dx, framework::Tensor* dy) {
  dim3 block_size = dim3(PADDLE_CUDA_THREAD_SIZE, 1);
  auto size = x->numel();
  dim3 grid_size =
      dim3((size + PADDLE_CUDA_THREAD_SIZE - 1) / PADDLE_CUDA_THREAD_SIZE, 1);
  SimpleElemwiseAddGradCUDAKernel<
      T><<<grid_size, block_size, 0,
           ctx.template device_context<plat::CUDADeviceContext>().stream()>>>(
      dout->data<T>(), size, dx->mutable_data<T>(ctx.GetPlace()),
      dy->mutable_data<T>(ctx.GetPlace()));
}

inline static bool UseReduceFirstAxisRank1(const framework::DDim& dout_dims,
                                           const framework::DDim& x_dims,
                                           const framework::DDim& y_dims,
                                           const int axis) {
  int start_axis =
      (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

  if (y_dims[y_dims.size() - 1] == 1) {
    return false;
  }

  if (y_dims.size() > 1) {
    for (int i = 0; i < y_dims.size() - 1; ++i) {
      if (y_dims[i] != 1) {
        return false;
      }
    }
    return true;
  } else if (start_axis == x_dims.size() - 1) {
    return true;
  }
  return false;
}

inline static bool UseReduceFirstAxisRank2(const framework::DDim& dout_dims,
                                           const framework::DDim& x_dims,
                                           const framework::DDim& y_dims,
                                           const int axis) {
  int start_axis =
      (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

  if (y_dims.size() < 2 ||
      x_dims[x_dims.size() - 2] != y_dims[y_dims.size() - 2] ||
      x_dims[x_dims.size() - 1] != y_dims[y_dims.size() - 1]) {
    return false;
  }

  if (start_axis == x_dims.size() - 2) {
    return true;
  } else if (start_axis == 0) {
    for (int i = 0; i < y_dims.size() - 2; ++i) {
      if (y_dims[i] != 1) {
        return false;
      }
    }
    return true;
  }
  return false;
}

inline static bool UseReduceSecondAxisRank2(const framework::DDim& dout_dims,
                                            const framework::DDim& x_dims,
                                            const framework::DDim& y_dims,
                                            const int axis, int* start,
                                            int* end) {
  if (x_dims.size() != y_dims.size() || y_dims.size() < 3) {
    return false;
  }

  auto y_dims_vec = framework::vectorize(y_dims);
  auto start_iter = std::find(y_dims_vec.begin(), y_dims_vec.end(), 1);
  auto end_iter = std::find(y_dims_vec.rbegin(), y_dims_vec.rend(), 1);
  if (start_iter == y_dims_vec.end() || start_iter == y_dims_vec.end() - 1) {
    return false;
  } else {
    *start = std::distance(y_dims_vec.begin(), start_iter);
    *end = y_dims_vec.size() - 1 - std::distance(y_dims_vec.rbegin(), end_iter);
    for (int i = *start; i <= *end; ++i) {
      if (y_dims[i] != 1) {
        return false;
      }
    }
    return true;
  }
}

template <typename T, typename OP>
__global__ __launch_bounds__(1024) void ReduceFirstAixsKernel(
    const T* in, T* out, const int64_t num_rows, const int64_t num_cols, OP op,
    T init) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  T sum = init;
  if (row < num_rows && col < num_cols) sum = in[row * num_cols + col];

  __shared__ __align__(
      alignof(T)) char partial_sums_raw[WARPSIZE * (WARPSIZE + 1) * sizeof(T)];
  T* partial_sums = reinterpret_cast<T*>(partial_sums_raw);

  row += gridDim.y * blockDim.y;

  if (col < num_cols) {
    for (; row < num_rows; row += gridDim.y * blockDim.y) {
      sum = op(sum, in[row * num_cols + col]);
    }
  }

  partial_sums[threadIdx.x * (WARPSIZE + 1) + threadIdx.y] = sum;

  __syncthreads();

  if (threadIdx.y == 0 && col < num_cols) {
    T s = partial_sums[threadIdx.x * (WARPSIZE + 1)];

    const int numRowsThisBlock = min(static_cast<int64_t>(blockDim.y),
                                     num_rows - blockIdx.y * blockDim.y);

    for (int row = 1; row < numRowsThisBlock; ++row) {
      T t = partial_sums[threadIdx.x * (WARPSIZE + 1) + row];
      s = op(s, t);
    }

    out[col * gridDim.y + blockIdx.y] = s;
  }
}

template <typename DeviceContext, typename T>
static void ElemwiseYGradRank1CUDA(const framework::ExecutionContext& ctx,
                                   const framework::Tensor& dout,
                                   const int rows, const int cols,
                                   framework::Tensor* dx,
                                   framework::Tensor* dy) {
  dim3 block_dim(WARPSIZE, std::min(rows, 1024 / WARPSIZE));
  dim3 grid_dim((cols + (WARPSIZE - 1)) / WARPSIZE, 1, 1);

  if (dx) {
    dx->mutable_data<T>(ctx.GetPlace());
    framework::TensorCopy(
        dout, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), dx);
  }
  if (dy) {
    dy->mutable_data<T>(ctx.GetPlace());
    const T* dout_data = dout.data<T>();
    T* dy_data = dy->data<T>();
    auto stream = ctx.template device_context<DeviceContext>().stream();
    ReduceFirstAixsKernel<<<grid_dim, block_dim, 0, stream>>>(
        dout_data, dy_data, rows, cols, AddFunctor<T>(), static_cast<T>(0));
  }
}

template <typename T, typename OP>
__global__ __launch_bounds__(1024) void ReduceFirstOrSecondAxisKernel(
    const T* in, T* out, const int num_planes, const int num_rows,
    const int num_cols, OP op, T init) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int elems_per_plane = num_rows * num_cols;

  const int plane = gid / num_cols;
  const int col = gid % num_cols;

  if (plane >= num_planes) return;

  if (num_rows == 1) {
    out[plane * elems_per_plane + col] = in[plane * elems_per_plane + col];
    return;
  }

  T sum = op(in[plane * elems_per_plane + col],
             in[plane * elems_per_plane + num_cols + col]);
  for (int row = 2; row < num_rows; ++row) {
    sum = op(sum, in[plane * elems_per_plane + row * num_cols + col]);
  }

  out[plane * num_cols + col] = sum;
}

template <typename DeviceContext, typename T>
static void ElemwiseYGradRank2CUDA(const framework::ExecutionContext& ctx,
                                   const framework::Tensor& dout,
                                   const int planes, const int rows,
                                   const int cols, framework::Tensor* dx,
                                   framework::Tensor* dy) {
  int num_threads = 128;
  int num_blocks = (rows + num_threads - 1) / num_threads;

  if (planes != 1) {
    num_blocks = (planes * cols + num_threads - 1) / num_threads;
  }

  if (dx) {
    dx->mutable_data<T>(ctx.GetPlace());
    framework::TensorCopy(
        dout, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), dx);
  }
  if (dy) {
    dy->mutable_data<T>(ctx.GetPlace());
    const T* dout_data = dout.data<T>();
    T* dy_data = dy->data<T>();
    auto stream = ctx.template device_context<DeviceContext>().stream();
    ReduceFirstOrSecondAxisKernel<<<num_blocks, num_threads, 0, stream>>>(
        dout_data, dy_data, planes, rows, cols, AddFunctor<T>(),
        static_cast<T>(0));
  }
}

template <typename DeviceContext, typename T>
static bool ElemwiseGradUseReduce(const framework::ExecutionContext& ctx,
                                  const int axis, const framework::DDim x_dims,
                                  const framework::DDim y_dims,
                                  const framework::Tensor& dout,
                                  framework::Tensor* dx,
                                  framework::Tensor* dy) {
  int start = 0;
  int end = 0;
  auto x_dims_vec = framework::vectorize(x_dims);
  if (UseReduceFirstAxisRank1(dout.dims(), x_dims, y_dims, axis)) {
    int rows = std::accumulate(x_dims_vec.begin(), x_dims_vec.end() - 1, 1,
                               std::multiplies<int>());
    int cols = dx->dims()[dx->dims().size() - 1];
    if (cols > 512 && cols < 4096) {
      ElemwiseYGradRank1CUDA<DeviceContext, T>(ctx, dout, rows, cols, dx, dy);
      return true;
    }
  }

  if (UseReduceFirstAxisRank2(dout.dims(), x_dims, y_dims, axis)) {
    int rows = std::accumulate(x_dims_vec.begin(), x_dims_vec.end() - 2, 1,
                               std::multiplies<int>());
    int cols =
        dx->dims()[dx->dims().size() - 1] * dx->dims()[dx->dims().size() - 2];
    if (cols > 4096) {
      ElemwiseYGradRank2CUDA<DeviceContext, T>(ctx, dout, 1, rows, cols, dx,
                                               dy);
      return true;
    }
  }

  if (UseReduceSecondAxisRank2(dout.dims(), x_dims, y_dims, axis, &start,
                               &end)) {
    int planes = std::accumulate(x_dims_vec.begin(), x_dims_vec.begin() + start,
                                 1, std::multiplies<int>());
    int rows = std::accumulate(x_dims_vec.begin() + start,
                               x_dims_vec.begin() + end + 1, 1,
                               std::multiplies<int>());
    int cols = std::accumulate(x_dims_vec.begin() + end + 1, x_dims_vec.end(),
                               1, std::multiplies<int>());
    if (rows / (planes * cols) < 16) {
      ElemwiseYGradRank2CUDA<DeviceContext, T>(ctx, dout, planes, rows, cols,
                                               dx, dy);
      return true;
    }
  }

  return false;
}

template <typename T>
class ElementwiseAddGradKernel<platform::CUDADeviceContext, T>
    : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);

    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    // skip out
    auto* out = dout;
    int axis = ctx.Attr<int>("axis");

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
    } else if (dx && dy && (dx->dims() == dy->dims())) {
      ElementwiseAddGrad<platform::CUDADeviceContext, T>(ctx, x, y, out, dout,
                                                         dx, dy);
    } else if (dx && dx->dims() == dout->dims() &&
               ElemwiseGradUseReduce<platform::CUDADeviceContext, T>(
                   ctx, axis, x->dims(), y->dims(), *dout, dx, dy)) {
    } else if (dy && dy->dims() == dout->dims() &&
               ElemwiseGradUseReduce<platform::CUDADeviceContext, T>(
                   ctx, axis, x->dims(), y->dims(), *dout, dy, dx)) {
    } else {
      DefaultElementwiseAddGrad<platform::CUDADeviceContext, T>(ctx, x, y, out,
                                                                dout, dx, dy);
    }
  }
};

}  // namespace operators
}  // namespace paddle
REGISTER_OP_CUDA_KERNEL(
    elementwise_add, ops::ElementwiseAddKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::complex64>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::complex128>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, plat::complex64>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, plat::complex128>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad_grad,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex64>,
    ops::ElementwiseAddDoubleGradKernel<plat::CUDADeviceContext,
                                        plat::complex128>);

REGISTER_OP_CUDA_KERNEL(
    grad_add, ops::ElementwiseAddKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::float16>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::complex64>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::complex128>);
