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
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<T>(ctx, &ins, &outs);
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        cuda_ctx, ins, &outs, axis, MaxFunctor<T>());
  }
};

template <typename T>
struct GreaterThanTFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    return a > b ? static_cast<T>(1) : static_cast<T>(0);
  }
};
template <typename T>
struct LessEqualThanTFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    return (a < b || a == b) ? static_cast<T>(1) : static_cast<T>(0);
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
  const auto& cuda_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();
  // dx
  if (dx != nullptr) {
    if (dx->dims() == dout->dims()) {
      // dx = dout * (x > y)
      framework::Tensor x_y_compare;
      x_y_compare.mutable_data<T>(dout->dims(), ctx.GetPlace());

      std::vector<const framework::Tensor*> ins = {x, y};
      std::vector<framework::Tensor*> outs = {&x_y_compare};
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          cuda_ctx, ins, &outs, axis, GreaterThanTFunctor<T>());
      default_elementwise_mul<DeviceContext, T>(ctx, dout, &x_y_compare, dx);
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(*dout)) {
        dx->clear();
        dx->mutable_data<T>(x->dims(), ctx.GetPlace());
      }
      std::vector<int> reduce_dims = GetReduceDim(x->dims(), out->dims(), axis);
      gpuStream_t stream = ctx.cuda_device_context().stream();

      framework::Tensor dx_tmp;
      dx_tmp.Resize(dout->dims());
      framework::Tensor x_y_compare;
      x_y_compare.mutable_data<T>(dout->dims(), ctx.GetPlace());

      std::vector<const framework::Tensor*> ins = {x, y};
      std::vector<framework::Tensor*> outs = {&x_y_compare};
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          cuda_ctx, ins, &outs, axis, GreaterThanTFunctor<T>());
      // dx(dx_tmp)=dout * y(x_y_compare)
      default_elementwise_mul<DeviceContext, T>(ctx, dout, &x_y_compare,
                                                &dx_tmp);
      TensorReduceFunctorImpl<T, T, CustomSum>(dx_tmp, dx, reduce_dims, stream);
    }
  }
  // dy
  if (dy != nullptr) {
    if (dy->dims() == dout->dims()) {
      // dy = dout * (x <= y)
      framework::Tensor x_y_compare;
      x_y_compare.mutable_data<T>(dout->dims(), ctx.GetPlace());

      std::vector<const framework::Tensor*> ins = {x, y};
      std::vector<framework::Tensor*> outs = {&x_y_compare};
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          cuda_ctx, ins, &outs, axis, LessEqualThanTFunctor<T>());

      default_elementwise_mul<DeviceContext, T>(ctx, dout, &x_y_compare, dy);
    } else {
      std::vector<int> reduce_dims = GetReduceDim(y->dims(), out->dims(), axis);
      gpuStream_t stream = ctx.cuda_device_context().stream();

      framework::Tensor dy_tmp;
      dy_tmp.Resize(dout->dims());
      framework::Tensor x_y_compare;
      x_y_compare.mutable_data<T>(dout->dims(), ctx.GetPlace());

      std::vector<const framework::Tensor*> ins = {x, y};
      std::vector<framework::Tensor*> outs = {&x_y_compare};
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          cuda_ctx, ins, &outs, axis, LessEqualThanTFunctor<T>());

      // dy(dy_tmp)=dout * x(x_y_compare)
      default_elementwise_mul<DeviceContext, T>(ctx, dout, &x_y_compare,
                                                &dy_tmp);
      TensorReduceFunctorImpl<T, T, CustomSum>(dy_tmp, dy, reduce_dims, stream);
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
