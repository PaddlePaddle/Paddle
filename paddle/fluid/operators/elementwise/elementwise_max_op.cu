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

#include "paddle/fluid/operators/elementwise/elementwise_max_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseMaxKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& dev_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, axis, MaxFunctor<T>());
  }
};

template <typename DeviceContext, typename T>
void DefaultElementMaxGrad(const framework::ExecutionContext& ctx,
                           const framework::Tensor* x,
                           const framework::Tensor* y,
                           const framework::Tensor* out,
                           const framework::Tensor* dout, framework::Tensor* dx,
                           framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();

  if (dx != nullptr) {
    // dx = dout * (x > y)
    framework::Tensor dx_result;
    dx_result.mutable_data<T>(dout->dims(), ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {x, y, dout};
    std::vector<framework::Tensor*> outs = {&dx_result};
    LaunchElementwiseCudaKernel<ElementwiseType::kTernary, T, T>(
        dev_ctx, ins, &outs, axis, TernaryGreaterThanFunctor<T>());

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
    // dy = dout * (x <= y)
    framework::Tensor dy_result;
    dy_result.mutable_data<T>(dout->dims(), ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {x, y, dout};
    std::vector<framework::Tensor*> outs = {&dy_result};
    LaunchElementwiseCudaKernel<ElementwiseType::kTernary, T, T>(
        dev_ctx, ins, &outs, axis, TernaryLessEqualThanFunctor<T>());

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

template <typename T>
class ElementwiseMaxGradKernel<platform::CUDADeviceContext, T>
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
    DefaultElementMaxGrad<platform::CUDADeviceContext, T>(ctx, x, y, out, dout,
                                                          dx, dy);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    elementwise_max,
    ops::ElementwiseMaxKernel<paddle::platform::CUDADeviceContext,
                              paddle::platform::float16>,
    ops::ElementwiseMaxKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseMaxKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseMaxKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseMaxKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_max_grad,
    ops::ElementwiseMaxGradKernel<paddle::platform::CUDADeviceContext,
                                  paddle::platform::float16>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);

REGISTER_OP_CUDA_KERNEL(
    elementwise_fmax,
    ops::ElementwiseFMaxKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseFMaxKernel<paddle::platform::CUDADeviceContext,
                               paddle::platform::float16>,
    ops::ElementwiseFMaxKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseFMaxKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseFMaxKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_fmax_grad,
    ops::ElementwiseFMaxGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseFMaxGradKernel<paddle::platform::CUDADeviceContext,
                                   paddle::platform::float16>,
    ops::ElementwiseFMaxGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseFMaxGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseFMaxGradKernel<paddle::platform::CUDADeviceContext,
                                   int64_t>);
