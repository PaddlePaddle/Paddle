/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

#include "paddle/fluid/operators/elementwise/elementwise_xpu.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseAddXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    XPUElementwise<T, XPUType>(ctx, xpu::broadcast_add<XPUType>);
  }
};

template <typename T>
class ElementwiseAddGradXPUKernel : public ElemwiseGradKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* dz = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    const framework::DDim& dz_dims = dz->dims();
    int axis = ctx.Attr<int>("axis");

    const T* dz_data = dz->data<T>();
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();

    if (dx != nullptr) {
      T* dx_data = dx->mutable_data<T>(ctx.GetPlace());
      if (dx->dims() == dz_dims) {
        if (dx_data != dz_data) {
          framework::TensorCopy(
              *dz, ctx.GetPlace(),
              ctx.template device_context<platform::DeviceContext>(), dx);
        }
      } else {
        // For inplace strategy, dx will be stored in addr of dz, which makes
        // the result of dy wrong.
        if (dx->IsSharedBufferWith(*dz)) {
          dx->clear();
          dx->mutable_data<T>(x->dims(), ctx.GetPlace());
        }
        std::vector<int> reduce_dims = GetReduceDim(dx->dims(), dz_dims, axis);
        std::vector<int> dz_vector = phi::vectorize<int>(dz_dims);

        int ret = xpu::reduce_sum<XPUType>(
            dev_ctx.x_context(), reinterpret_cast<const XPUType*>(dz_data),
            reinterpret_cast<XPUType*>(dx_data), dz_vector, reduce_dims);
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_sum");
      }
    }

    if (dy != nullptr) {
      T* dy_data = dy->mutable_data<T>(ctx.GetPlace());
      if (dy->dims() == dz_dims) {
        if (dy_data != dz_data) {
          framework::TensorCopy(
              *dz, ctx.GetPlace(),
              ctx.template device_context<platform::DeviceContext>(), dy);
        }
      } else {
        std::vector<int> reduce_dims = GetReduceDim(dy->dims(), dz_dims, axis);
        std::vector<int> dz_vector = phi::vectorize<int>(dz_dims);
        int ret = xpu::reduce_sum<XPUType>(
            dev_ctx.x_context(), reinterpret_cast<const XPUType*>(dz_data),
            reinterpret_cast<XPUType*>(dy_data), dz_vector, reduce_dims);
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_sum");
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(elementwise_add, ops::ElementwiseAddXPUKernel<float>,
                       ops::ElementwiseAddXPUKernel<paddle::platform::float16>);
REGISTER_OP_XPU_KERNEL(
    elementwise_add_grad, ops::ElementwiseAddGradXPUKernel<float>,
    ops::ElementwiseAddGradXPUKernel<paddle::platform::float16>);
#endif
