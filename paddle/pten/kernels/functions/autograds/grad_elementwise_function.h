/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/cpu/utils.h"
#include "paddle/pten/kernels/functions/autograds/grad_elementwise_function.cu.h"
#include "paddle/pten/kernels/functions/eigen/common.h"
#include "paddle/pten/kernels/functions/general/elementwise_base.h"
#include "paddle/pten/kernels/functions/general/transform_function.h"

namespace pten {
namespace math {

template <typename T>
struct IdentityGrad {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename T, typename DX_OP, typename DY_OP>
void CommonGradBroadcastCPU(const DenseTensor &x,
                            const DenseTensor &y,
                            const DenseTensor &out,
                            const DenseTensor &dout,
                            DenseTensor *dx,
                            DenseTensor *dy,
                            int *x_dims_array,
                            int *y_dims_array,
                            int *out_dims_array,
                            int max_dim,
                            const paddle::platform::CPUDeviceContext &ctx,
                            DX_OP dx_op,
                            DY_OP dy_op) {
  std::vector<int> index_array(max_dim, 0);
  const T *x_data = x.data<T>();
  const T *y_data = y.data<T>();
  const T *out_data = out.data<T>();
  const T *dout_data = dout.data<T>();
  T *dx_data = dx == nullptr ? nullptr : dx->mutable_data<T>();
  T *dy_data = dy == nullptr ? nullptr : dy->mutable_data<T>();
  if (dx_data != nullptr) {
    memset(dx_data, 0, dx->numel() * sizeof(T));
  }
  if (dy_data != nullptr) {
    memset(dy_data, 0, dy->numel() * sizeof(T));
  }
  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_index, y_index;
  for (int out_index = 0; out_index < out_size; ++out_index) {
    x_index = paddle::operators::GetElementwiseIndex(
        x_dims_array, max_dim, index_array.data());
    y_index = paddle::operators::GetElementwiseIndex(
        y_dims_array, max_dim, index_array.data());
    if (dx_data != nullptr) {
      dx_data[x_index] += dx_op(x_data[x_index],
                                y_data[y_index],
                                out_data[out_index],
                                dout_data[out_index]);
    }
    if (dy_data != nullptr) {
      dy_data[y_index] += dy_op(x_data[x_index],
                                y_data[y_index],
                                out_data[out_index],
                                dout_data[out_index]);
    }

    paddle::operators::UpdateElementwiseIndexArray(
        out_dims_array, max_dim, index_array.data());
  }
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void CommonElementwiseBroadcastBackward(const DeviceContext &ctx,
                                        const paddle::framework::DDim &x_dims,
                                        const paddle::framework::DDim &y_dims,
                                        const DenseTensor &x,
                                        const DenseTensor &y,
                                        const DenseTensor &out,
                                        const DenseTensor &dout,
                                        int axis,
                                        DenseTensor *dx,
                                        DenseTensor *dy,
                                        DX_OP dx_op,
                                        DY_OP dy_op) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  paddle::operators::GetBroadcastDimsArrays(x_dims,
                                            y_dims,
                                            x_dims_array.data(),
                                            y_dims_array.data(),
                                            out_dims_array.data(),
                                            max_dim,
                                            axis);
  // for inplace strategy. memset will make dx and dout clear and get wrong
  // result.

  /* !!! Skip special case for inplace for now !!!
  if (dx && dx->IsSharedBufferWith(dout)) {
    dx->clear();
    dx->mutable_data<T>(x_dims, ctx.GetPlace());
  }
  */

  VLOG(3) << "CommonElementwiseBroadcastBackward xdims:"
          << paddle::framework::make_ddim(x_dims_array)
          << " ydim:" << paddle::framework::make_ddim(y_dims_array);

  if (paddle::platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
    CommonGradBroadcastCUDA<T, DX_OP, DY_OP>(x,
                                             y,
                                             out,
                                             dout,
                                             dx,
                                             dy,
                                             x_dims_array.data(),
                                             y_dims_array.data(),
                                             out_dims_array.data(),
                                             max_dim,
                                             ctx,
                                             dx_op,
                                             dy_op);
#endif
  } else {
    CommonGradBroadcastCPU<T, DX_OP, DY_OP>(x,
                                            y,
                                            out,
                                            dout,
                                            dx,
                                            dy,
                                            x_dims_array.data(),
                                            y_dims_array.data(),
                                            out_dims_array.data(),
                                            max_dim,
                                            ctx,
                                            dx_op,
                                            dy_op);
  }
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseGradComputeNoBroadcast(const DeviceContext &ctx,
                                    const paddle::framework::DDim &x_dim,
                                    const paddle::framework::DDim &y_dim,
                                    const DenseTensor &x,
                                    const DenseTensor &y,
                                    const DenseTensor &out,
                                    const DenseTensor &dout,
                                    int axis,
                                    DenseTensor *dx,
                                    DenseTensor *dy,
                                    DX_OP dx_op,
                                    DY_OP dy_op) {
  size_t N = static_cast<size_t>(paddle::framework::product(x_dim));
#if !defined(_WIN32)
  paddle::platform::ForRange<DeviceContext> for_range(ctx, N);
#else
  paddle::platform::ForRange<DeviceContext> for_range(ctx, N);
#endif  // !_WIN32
  for_range(paddle::operators::ElemwiseGradNoBroadcast<T, DX_OP, DY_OP>{
      x.data<T>(),
      y.data<T>(),
      out.data<T>(),
      dout.data<T>(),
      dx_op,
      dy_op,
      dx == nullptr ? nullptr : dx->mutable_data<T>(),
      dy == nullptr ? nullptr : dy->mutable_data<T>()});
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseGradComputeWithBroadcast(const DeviceContext &ctx,
                                      const paddle::framework::DDim &x_dims,
                                      const paddle::framework::DDim &y_dims,
                                      const DenseTensor &x,
                                      const DenseTensor &y,
                                      const DenseTensor &out,
                                      const DenseTensor &dout,
                                      int axis,
                                      DenseTensor *dx,
                                      DenseTensor *dy,
                                      DX_OP dx_op,
                                      DY_OP dy_op) {
  bool is_xsize_larger = true;

  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      paddle::platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    paddle::platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimed = paddle::operators::trim_trailing_singular_dims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    pten::general::get_mid_dims(x_dims,
                                y_dims_trimed,
                                axis_trim,
                                &pre,
                                &n,
                                &post,
                                &is_run_common_broadcast);
  } else {
    auto x_dims_trimed = paddle::operators::trim_trailing_singular_dims(x_dims);
    axis_trim = (x_dims_trimed.size() == 0) ? y_dims.size() : axis;
    pten::general::get_mid_dims(y_dims,
                                x_dims_trimed,
                                axis_trim,
                                &pre,
                                &n,
                                &post,
                                &is_run_common_broadcast);
  }
  // special case for common backward implementation.
  if (is_run_common_broadcast) {
    CommonElementwiseBroadcastBackward<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dims, y_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
    return;
  }
  if (post == 1) {
    if (paddle::platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      paddle::operators::ElemwiseGradBroadcast1CUDA(
          ctx.stream(),
          x.data<T>(),
          y.data<T>(),
          out.data<T>(),
          dout.data<T>(),
          pre,
          n,
          is_xsize_larger,
          dx_op,
          dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(),
          dy == nullptr ? nullptr : dy->mutable_data<T>());
#endif
    } else {
      paddle::operators::ElemwiseGradBroadcast1CPU(
          x.data<T>(),
          y.data<T>(),
          out.data<T>(),
          dout.data<T>(),
          pre,
          n,
          is_xsize_larger,
          dx_op,
          dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(),
          dy == nullptr ? nullptr : dy->mutable_data<T>());
    }
  } else {
    if (paddle::platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
      paddle::operators::ElemwiseGradBroadcast2CUDA(
          ctx.stream(),
          x.data<T>(),
          y.data<T>(),
          out.data<T>(),
          dout.data<T>(),
          pre,
          n,
          post,
          is_xsize_larger,
          dx_op,
          dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(),
          dy == nullptr ? nullptr : dy->mutable_data<T>());
#endif
    } else {
      paddle::operators::ElemwiseGradBroadcast2CPU(
          x.data<T>(),
          y.data<T>(),
          out.data<T>(),
          dout.data<T>(),
          pre,
          n,
          post,
          is_xsize_larger,
          dx_op,
          dy_op,
          dx == nullptr ? nullptr : dx->mutable_data<T>(),
          dy == nullptr ? nullptr : dy->mutable_data<T>());
    }
  }
}

template <typename DeviceContext, typename T, typename DX_OP, typename DY_OP>
void ElemwiseExplicitGradCompute(const DeviceContext &ctx,
                                 const DenseTensor &x,
                                 const DenseTensor &y,
                                 const DenseTensor &out,
                                 const DenseTensor &dout,
                                 int axis,
                                 DenseTensor *dx,
                                 DenseTensor *dy,
                                 DX_OP dx_op,
                                 DY_OP dy_op) {
  const paddle::framework::DDim &x_dim = x.dims();
  const paddle::framework::DDim &y_dim = y.dims();
  if (x.dims() == y.dims()) {
    ElemwiseGradComputeNoBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dim, y_dim, dout, dout, out, dout, axis, dx, dy, dx_op, dy_op);
  } else {
    ElemwiseGradComputeWithBroadcast<DeviceContext, T, DX_OP, DY_OP>(
        ctx, x_dim, y_dim, dout, dout, out, dout, axis, dx, dy, dx_op, dy_op);
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext,
                 paddle::platform::CPUDeviceContext>::value>::type
default_elementwise_add_grad(const DeviceContext &ctx,
                             const DenseTensor &x,
                             const DenseTensor &y,
                             const DenseTensor &out,
                             const DenseTensor &dout,
                             int axis,
                             DenseTensor *dx,
                             DenseTensor *dy) {
  ElemwiseExplicitGradCompute<DeviceContext,
                              T,
                              IdentityGrad<T>,
                              IdentityGrad<T>>(
      ctx, x, y, out, dout, axis, dx, dy, IdentityGrad<T>(), IdentityGrad<T>());
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_floating_point<T>::value &&
    std::is_same<DeviceContext,
                 paddle::platform::CPUDeviceContext>::value>::type
elementwise_add_grad(const DeviceContext &ctx,
                     const DenseTensor &x,
                     const DenseTensor &y,
                     const DenseTensor &out,
                     const DenseTensor &dout,
                     int axis,
                     DenseTensor *dx,
                     DenseTensor *dy) {
  auto blas = paddle::operators::math::GetBlas<DeviceContext, T>(ctx);
  if (dx) {
    blas.VCOPY(dout.numel(), dout.data<T>(), dx->mutable_data<T>());
  }

  if (dy) {
    blas.VCOPY(dout.numel(), dout.data<T>(), dy->mutable_data<T>());
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    !std::is_floating_point<T>::value &&
    std::is_same<DeviceContext,
                 paddle::platform::CPUDeviceContext>::value>::type
elementwise_add_grad(const DeviceContext &ctx,
                     const DenseTensor &x,
                     const DenseTensor &y,
                     const DenseTensor &out,
                     const DenseTensor &dout,
                     int axis,
                     DenseTensor *dx,
                     DenseTensor *dy) {
  default_elementwise_add_grad<DeviceContext, T>(
      ctx, x, y, out, dout, axis, dx, dy);
}

template <typename T>
class ElemwiseGradKernel {
 public:
  void Compute(const DeviceContext &dev_ctx,
               const DenseTensor &GradOut,
               DenseTensor *GradX) const {
    if (GradX != nullptr) {
      GradX->set_lod(GradOut.lod());
    }
  }
};

template <typename DeviceContext, typename T>
void ElementwiseAddGradFunction(const DeviceContext &dev_ctx,
                                const DenseTensor &X,
                                const DenseTensor &Y,
                                const DenseTensor &GradOut,
                                int axis,
                                DenseTensor *GradX,
                                DenseTensor *GradY) {
  ElemwiseGradKernel<T>().Compute(dev_ctx, GradOut, GradX);

  // Special case when dy is not needed and dx doesn't reduce
  if (GradX != nullptr && GradY == nullptr && GradX->dims() == GradOut.dims()) {
    VLOG(4) << "Special case when dy is not needed and dx doesn't "
               "reduce";
    pten::Copy(dev_ctx, GradOut, GradX);

  } else if (GradX == nullptr && GradY != nullptr &&
             GradY->dims() == GradOut.dims()) {
    VLOG(4) << "Special case when dx is not needed and dy doesn't "
               "reduce";
    pten::Copy(dev_ctx, GradOut, GradY);

  } else if (GradX != nullptr && GradY != nullptr &&
             (GradX->dims() == GradY->dims())) {
    elementwise_add_grad<DeviceContext, T>(
        dev_ctx, X, Y, GradOut, GradOut, axis, GradX, GradY);

  } else {
    default_elementwise_add_grad<DeviceContext, T>(
        dev_ctx, X, Y, GradOut, GradOut, axis, GradX, GradY);
  }
}

}  // namespace math
}  // namespace pten
