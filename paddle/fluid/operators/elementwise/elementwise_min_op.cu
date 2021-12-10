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
#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
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

template <typename T>
struct LessThanFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    return a < b ? static_cast<T>(1) : static_cast<T>(0);
  }
};
template <typename T>
struct GreaterEqualThanFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    return (a > b || a == b) ? static_cast<T>(1) : static_cast<T>(0);
  }
};

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
  // dx
  if (dx != nullptr) {
    // For inplace strategy, dx will be stored in addr of dout, which makes
    // the result of dy wrong.
    if (dx->IsSharedBufferWith(*dout)) {
      dx->clear();
      dx->mutable_data<T>(x->dims(), ctx.GetPlace());
    }

    framework::Tensor compare_xy;
    compare_xy.mutable_data<T>(dout->dims(), ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {x, y};
    std::vector<framework::Tensor*> outs = {&compare_xy};
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, axis, LessThanFunctor<T>());

    if (dx->dims() == dout->dims()) {
      // dx = dout * (x < y)
      default_elementwise_mul<DeviceContext, T>(ctx, dout, &compare_xy, dx);
    } else {
      std::vector<int> reduce_dims = GetReduceDim(x->dims(), out->dims(), axis);
      gpuStream_t stream = ctx.cuda_device_context().stream();
      framework::Tensor dx_tmp;
      dx_tmp.Resize(dout->dims());

      // dx(dx_tmp)=dout * y(compare_xy)
      default_elementwise_mul<DeviceContext, T>(ctx, dout, &compare_xy,
                                                &dx_tmp);
      TensorReduceFunctorImpl<T, T, CustomSum>(dx_tmp, dx, reduce_dims, stream);
    }
  }
  // dy
  if (dy != nullptr) {
    framework::Tensor compare_xy;
    compare_xy.mutable_data<T>(dout->dims(), ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {x, y};
    std::vector<framework::Tensor*> outs = {&compare_xy};
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
        dev_ctx, ins, &outs, axis, GreaterEqualThanFunctor<T>());
    if (dy->dims() == dout->dims()) {
      // dy = dout * (x >= y);
      default_elementwise_mul<DeviceContext, T>(ctx, dout, &compare_xy, dy);
    } else {
      std::vector<int> reduce_dims = GetReduceDim(y->dims(), out->dims(), axis);
      gpuStream_t stream = ctx.cuda_device_context().stream();
      framework::Tensor dy_tmp;
      dy_tmp.Resize(dout->dims());

      // dy(dy_tmp)=dout * x(compare_xy)
      default_elementwise_mul<DeviceContext, T>(ctx, dout, &compare_xy,
                                                &dy_tmp);
      TensorReduceFunctorImpl<T, T, CustomSum>(dy_tmp, dy, reduce_dims, stream);
    }
  }
}

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
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseMinKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_min_grad,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ElementwiseMinGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
