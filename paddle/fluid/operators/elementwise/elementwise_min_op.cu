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

#include "paddle/fluid/operators/elementwise/elementwise_min_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseMinKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& dev_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, axis, MinFunctor<T>());
  }
};

template <typename InT, typename OutT>
struct MinGradXYFunctor {
  inline HOSTDEVICE paddle::framework::Array<OutT, 2> operator()(
      const InT& a,    // x
      const InT& b,    // y
      const InT& c) {  // dout
    paddle::framework::Array<OutT, 2> outs;
    // dx = dout * (x < y)
    outs[0] = a < b ? c : static_cast<InT>(0);
    // dy = dout * (x >= y)
    outs[1] = (a > b || a == b) ? c : static_cast<InT>(0);
    return outs;
  }
};

template <typename T>
void ReduceWrapper(const platform::CUDADeviceContext& dev_ctx, int axis,
                   const framework::Tensor* in, const framework::Tensor* out,
                   framework::Tensor* src, framework::Tensor* dst) {
  std::vector<int> reduce_dims = GetReduceDim(in->dims(), out->dims(), axis);
  TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
      *src, dst, kps::IdentityFunctor<T>(), reduce_dims, dev_ctx.stream());
}

template <typename DeviceContext, typename T>
void DefaultElementMinGrad(const framework::ExecutionContext& ctx,
                           const framework::Tensor* x,
                           const framework::Tensor* y,
                           const framework::Tensor* out,
                           const framework::Tensor* dout, framework::Tensor* dx,
                           framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();
  framework::Tensor tmp_dx;
  framework::Tensor tmp_dy;
  tmp_dx.mutable_data<T>(dout->dims(), ctx.GetPlace());
  tmp_dy.mutable_data<T>(dout->dims(), ctx.GetPlace());

  if (dx != nullptr && dy != nullptr) {
    dx->mutable_data<T>(ctx.GetPlace());
    dy->mutable_data<T>(ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {x, y, dout};
    std::vector<framework::Tensor*> outs;
    if (dx->dims() == dout->dims() && dy->dims() == dout->dims()) {
      outs = {dx, dy};
    } else if (dx->dims() != dout->dims() && dy->dims() == dout->dims()) {
      outs = {&tmp_dx, dy};
    } else if (dx->dims() == dout->dims() && dy->dims() != dout->dims()) {
      outs = {dx, &tmp_dy};
    } else if (dx->dims() != dout->dims() && dy->dims() != dout->dims()) {
      outs = {&tmp_dx, &tmp_dy};
    }
    auto functor = MinGradXYFunctor<T, T>();
    LaunchElementwiseCudaKernel<ElementwiseType::kTernary, T, T,
                                decltype(functor), 2>(dev_ctx, ins, &outs, axis,
                                                      functor);
    if (dx->dims() != dout->dims() && dy->dims() == dout->dims()) {
      ReduceWrapper<T>(dev_ctx, axis, x, out, &tmp_dx, dx);
    } else if (dx->dims() == dout->dims() && dy->dims() != dout->dims()) {
      ReduceWrapper<T>(dev_ctx, axis, y, out, &tmp_dy, dy);
    } else if (dx->dims() != dout->dims() && dy->dims() != dout->dims()) {
      ReduceWrapper<T>(dev_ctx, axis, x, out, &tmp_dx, dx);
      ReduceWrapper<T>(dev_ctx, axis, y, out, &tmp_dy, dy);
    }

  } else if (dx != nullptr && dy == nullptr) {
    dx->mutable_data<T>(ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {x, y, dout};
    std::vector<framework::Tensor*> outs;
    if (dx->dims() != dout->dims()) {
      outs = {&tmp_dx};
    } else {
      outs = {dx};
    }

    LaunchElementwiseCudaKernel<ElementwiseType::kTernary, T, T>(
        dev_ctx, ins, &outs, axis, TernaryLessThanFunctor<T>());
    if (dx->dims() != dout->dims()) {
      ReduceWrapper<T>(dev_ctx, axis, x, out, &tmp_dx, dx);
    }
  } else if (dx == nullptr && dy != nullptr) {
    dy->mutable_data<T>(ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {x, y, dout};
    std::vector<framework::Tensor*> outs;
    if (dy->dims() != dout->dims()) {
      outs = {&tmp_dy};
    } else {
      outs = {dy};
    }

    LaunchElementwiseCudaKernel<ElementwiseType::kTernary, T, T>(
        dev_ctx, ins, &outs, axis, TernaryGreaterEqualThanFunctor<T>());
    if (dy->dims() != dout->dims()) {
      ReduceWrapper<T>(dev_ctx, axis, y, out, &tmp_dy, dy);
    }
  }
}

/*
template <typename DeviceContext, typename T>
void DefaultElementMinGrad(const framework::ExecutionContext& ctx,
                           const framework::Tensor* x,
                           const framework::Tensor* y,
                           const framework::Tensor* out,
                           const framework::Tensor* dout, framework::Tensor* dx,
                           framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();

  if (dx != nullptr) {
    // dx = dout * (x < y)
    framework::Tensor dx_result;
    dx_result.mutable_data<T>(dout->dims(), ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {x, y, dout};
    std::vector<framework::Tensor*> outs = {&dx_result};
    LaunchElementwiseCudaKernel<ElementwiseType::kTernary, T, T>(
        dev_ctx, ins, &outs, axis, TernaryLessThanFunctor<T>());

    if (dx->dims() == dout->dims()) {
      framework::TensorCopy(
          dx_result, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), dx);
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(*dout)) {
        dx->clear();
        dx->mutable_data<T>(x->dims(), ctx.GetPlace());
      }
      std::vector<int> reduce_dims = GetReduceDim(x->dims(), out->dims(), axis);
      gpuStream_t stream = ctx.cuda_device_context().stream();
      TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dx_result, dx, kps::IdentityFunctor<T>(), reduce_dims, stream);
    }
  }

  // dy
  if (dy != nullptr) {
    // dy = dout * (x >= y)
    framework::Tensor dy_result;
    dy_result.mutable_data<T>(dout->dims(), ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {x, y, dout};
    std::vector<framework::Tensor*> outs = {&dy_result};
    LaunchElementwiseCudaKernel<ElementwiseType::kTernary, T, T>(
        dev_ctx, ins, &outs, axis, TernaryGreaterEqualThanFunctor<T>());

    if (dy->dims() == dout->dims()) {
      framework::TensorCopy(
          dy_result, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), dy);
    } else {
      std::vector<int> reduce_dims = GetReduceDim(y->dims(), out->dims(), axis);
      gpuStream_t stream = ctx.cuda_device_context().stream();
      TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dy_result, dy, kps::IdentityFunctor<T>(), reduce_dims, stream);
    }
  }
}
*/

template <typename T>
class ElementwiseMinGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto* out = dout;  // Fake out, not used
    DefaultElementMinGrad<platform::CUDADeviceContext, T>(ctx, x, y, out, dout,
                                                          dx, dy);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    elementwise_min,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::float16>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_min_grad,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);

REGISTER_OP_CUDA_KERNEL(
    elementwise_fmin,
    ops::ElementwiseFMinKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseFMinKernel<paddle::platform::CUDADeviceContext,
                               paddle::platform::float16>,
    ops::ElementwiseFMinKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseFMinKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseFMinKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_fmin_grad,
    ops::ElementwiseFMinGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseFMinGradKernel<paddle::platform::CUDADeviceContext,
                                   paddle::platform::float16>,
    ops::ElementwiseFMinGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseFMinGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseFMinGradKernel<paddle::platform::CUDADeviceContext,
                                   int64_t>);
