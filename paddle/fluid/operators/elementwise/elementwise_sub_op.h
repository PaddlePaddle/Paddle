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
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
void default_elementwise_sub(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y, framework::Tensor* z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          SubFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseSubFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseSubFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T, class Enable = void>
struct SameDimsElemwiseSub {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z);
};

template <typename DeviceContext, typename T>
class ElementwiseSubKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());

    auto dims_equal = x->dims() == y->dims();
    if (dims_equal) {
      SameDimsElemwiseSub<DeviceContext, T> same_dims_sub;
      same_dims_sub(ctx, x, y, z);
    } else {
      default_elementwise_sub<DeviceContext, T>(ctx, x, y, z);
    }
  }
};

template <typename T>
struct SubGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename T>
struct SubGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return -dout; }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_sub_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  ElemwiseExplicitGradCompute<DeviceContext, T, SubGradDX<T>, SubGradDY<T>>(
      ctx, *x, *y, *out, *dout, axis, dx, dy, SubGradDX<T>(), SubGradDY<T>());
}

#ifdef PADDLE_WITH_CUDA
// cuda definition
template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
elementwise_sub_grad(const framework::ExecutionContext& ctx,
                     const framework::Tensor* x, const framework::Tensor* y,
                     const framework::Tensor* out,
                     const framework::Tensor* dout, framework::Tensor* dx,
                     framework::Tensor* dy);
#endif

static bool UseEigenBroadcast(const framework::DDim& x_dims,
                              const framework::DDim& y_dims) {
  int bcast_dims_remainder = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    if (x_dims[i] >= y_dims[i]) {
      bcast_dims_remainder += x_dims[i] % y_dims[i];
    } else {
      bcast_dims_remainder += y_dims[i] % x_dims[i];
    }
  }
  if (bcast_dims_remainder == 0) {
    return true;
  } else {
    return false;
  }
}

template <int Rank>
static void GetBraodcastDims(const framework::DDim& x_dims,
                             const framework::DDim& y_dims,
                             Eigen::DSizes<int, Rank>* x_bcast_dims,
                             Eigen::DSizes<int, Rank>* y_bcast_dims) {
  for (int i = 0; i < x_dims.size(); ++i) {
    if (x_dims[i] >= y_dims[i]) {
      (*x_bcast_dims)[i] = 1;
      (*y_bcast_dims)[i] = x_dims[i] / y_dims[i];
    } else {
      (*y_bcast_dims)[i] = 1;
      (*x_bcast_dims)[i] = y_dims[i] / x_dims[i];
    }
  }
}

static framework::DDim GetNewDims(const framework::DDim& in_dims, int rank) {
  std::vector<int64_t> new_dims_vec(rank);
  if (in_dims.size() < rank) {
    for (int i = 0; i < rank - in_dims.size(); ++i) {
      new_dims_vec[i] = 1;
    }
    for (int i = 0; i < in_dims.size(); ++i) {
      new_dims_vec[i + rank - in_dims.size()] = in_dims[i];
    }
  } else {
    new_dims_vec = vectorize(in_dims);
  }
  return framework::make_ddim(new_dims_vec);
}

template <typename DeviceContext, typename T, int Rank>
static void ElementwiseSubGradFunction(
    const framework::ExecutionContext& context) {
  auto dout = context.Input<framework::Tensor>(framework::GradVarName("Out"));
  auto dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
  auto dy = context.Output<framework::Tensor>(framework::GradVarName("Y"));

  auto x_dims = context.Input<framework::Tensor>("X")->dims();
  auto y_dims = context.Input<framework::Tensor>("Y")->dims();
  auto out_dims = dout->dims();

  auto dout_t = framework::EigenTensor<T, Rank>::From(*dout);

  framework::DDim x_new_dims = GetNewDims(x_dims, Rank);
  framework::DDim y_new_dims = GetNewDims(y_dims, Rank);

  Eigen::DSizes<int, Rank> x_bcast_dims;
  Eigen::DSizes<int, Rank> y_bcast_dims;

  GetBraodcastDims<Rank>(x_new_dims, y_new_dims, &x_bcast_dims, &y_bcast_dims);

  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  Eigen::DSizes<int, Rank * 2> x_reshape_dims;
  Eigen::DSizes<int, Rank * 2> y_reshape_dims;
  Eigen::DSizes<int, Rank> reduce_dims;
  for (int i = 0; i < x_new_dims.size(); ++i) {
    x_reshape_dims[2 * i] = x_bcast_dims[i];
    x_reshape_dims[2 * i + 1] = x_new_dims[i];
    y_reshape_dims[2 * i] = y_bcast_dims[i];
    y_reshape_dims[2 * i + 1] = y_new_dims[i];
    reduce_dims[i] = 2 * i;
  }
  VLOG(3) << "x_reshape_dims: " << x_reshape_dims;
  VLOG(3) << "y_reshape_dims: " << y_reshape_dims;
  VLOG(3) << "reduce_dims: " << reduce_dims;

  framework::Tensor dout_back;
  dout_back.Resize(dout->dims());
  dout_back.mutable_data<T>(context.GetPlace());
  auto dout_back_t = framework::EigenTensor<T, Rank>::From(dout_back);
  dout_back_t.device(place) = dout_t;
  if (dx) {
    dx->mutable_data<T>(context.GetPlace());
    auto dx_t = framework::EigenTensor<T, Rank>::From(*dx, x_new_dims);
    if (x_dims == out_dims) {
      dx_t.device(place) = dout_t;
    } else {
      dx_t.device(place) = dout_t.reshape(x_reshape_dims)
                               .sum(reduce_dims)
                               .reshape(dx_t.dimensions());
    }
  }
  if (dy) {
    dy->mutable_data<T>(context.GetPlace());
    auto dy_t = framework::EigenTensor<T, Rank>::From(*dy, y_new_dims);
    if (y_dims == out_dims) {
      dy_t.device(place) = -dout_back_t;
    } else {
      dy_t.device(place) = -dout_back_t.reshape(y_reshape_dims)
                                .sum(reduce_dims)
                                .reshape(dy_t.dimensions());
    }
  }
}

template <typename DeviceContext, typename T>
void ElementwiseSubGradEigenFunction(
    const framework::ExecutionContext& context) {
  auto x_rank = context.Input<framework::Tensor>("X")->dims().size();
  auto y_rank = context.Input<framework::Tensor>("Y")->dims().size();
  auto rank = std::max(x_rank, y_rank);

  switch (rank) {
    case 1:
      ElementwiseSubGradFunction<DeviceContext, T, 1>(context);
      break;
    case 2:
      ElementwiseSubGradFunction<DeviceContext, T, 2>(context);
      break;
    case 3:
      ElementwiseSubGradFunction<DeviceContext, T, 3>(context);
      break;
    case 4:
      ElementwiseSubGradFunction<DeviceContext, T, 4>(context);
      break;
    case 5:
      ElementwiseSubGradFunction<DeviceContext, T, 5>(context);
      break;
    case 6:
      ElementwiseSubGradFunction<DeviceContext, T, 6>(context);
      break;
  }
}

template <typename DeviceContext, typename T>
class ElementwiseSubGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    auto rank = std::max(x_dims.size(), y_dims.size());
    framework::DDim x_new_dims = GetNewDims(x_dims, rank);
    framework::DDim y_new_dims = GetNewDims(y_dims, rank);
    bool use_eigen = UseEigenBroadcast(x_new_dims, y_new_dims);
    if (use_eigen) {
      VLOG(3) << "====ues eigen grad function====";
      ElementwiseSubGradEigenFunction<DeviceContext, T>(ctx);
      return;
    }
    // skip out
    auto* out = dout;
    if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
      elementwise_sub_grad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
    } else {
      ElemwiseExplicitGradCompute<DeviceContext, T, SubGradDX<T>, SubGradDY<T>>(
          ctx, *x, *y, *out, *dout, axis, dx, dy, SubGradDX<T>(),
          SubGradDY<T>());
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseSubDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>("DOut");
    auto* ddx = ctx.Input<Tensor>("DDX");
    auto* ddy = ctx.Input<Tensor>("DDY");

    auto* ddout = ctx.Output<Tensor>("DDOut");

    // DDOut = ddx - ddy
    if (ddout) {
      Tensor ddx_safe, ddy_safe;
      GetDoubleGradSafeTensor<DeviceContext, T>(ctx, dout, ddx, &ddx_safe);
      GetDoubleGradSafeTensor<DeviceContext, T>(ctx, y, ddy, &ddy_safe);

      ddout->mutable_data<T>(ctx.GetPlace());
      int axis = ctx.Attr<int>("axis");
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &ddx_safe, &ddy_safe, axis, SubFunctor<T>(), ddout);
    }
  }
};

}  // namespace operators
}  // namespace paddle
