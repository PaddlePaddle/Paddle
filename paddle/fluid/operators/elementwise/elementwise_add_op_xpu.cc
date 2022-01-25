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

static std::vector<int> get_rdims(const std::vector<int>& xdims,
                                  const std::vector<int>& ydims) {
  std::vector<int> rdims;
  for (size_t i = 0; i < xdims.size(); i++) {
    if (xdims[i] != ydims[i]) {
      rdims.push_back(i);
    }
  }
  return rdims;
}

template <typename T>
class ElementwiseAddGradXPUKernel : public ElemwiseGradKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dz = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    const framework::DDim& x_dims = x->dims();
    const framework::DDim& y_dims = y->dims();
    const framework::DDim& dz_dims = dz->dims();
    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
    int max_dim = std::max(x_dims.size(), y_dims.size());
    PADDLE_ENFORCE_GE(
        axis, 0,
        platform::errors::InvalidArgument(
            "Axis should be great than or equal to 0, but received axis is %d.",
            axis));
    PADDLE_ENFORCE_LT(
        axis, max_dim,
        platform::errors::InvalidArgument(
            "Axis should be less than %d, but received axis is %d.", max_dim,
            axis));

    std::vector<int> x_dims_vec(max_dim, 1);
    std::vector<int> y_dims_vec(max_dim, 1);
    std::vector<int> z_dims_vec(max_dim, 1);
    if (x_dims.size() == max_dim) {
      for (int i = 0; i < max_dim; i++) {
        x_dims_vec[i] = x_dims[i];
      }
    } else {
      for (int i = 0; i < x_dims.size(); i++) {
        x_dims_vec[i + axis] = x_dims[i];
      }
    }

    if (y_dims.size() == max_dim) {
      for (int i = 0; i < max_dim; i++) {
        y_dims_vec[i] = y_dims[i];
      }
    } else {
      for (int i = 0; i < y_dims.size(); i++) {
        y_dims_vec[i + axis] = y_dims[i];
      }
    }

    for (int i = 0; i < max_dim; i++) {
      z_dims_vec[i] = dz_dims[i];
    }
    std::vector<int> rdims_for_x;
    std::vector<int> rdims_for_y;
    rdims_for_x = get_rdims(x_dims_vec, z_dims_vec);
    rdims_for_y = get_rdims(y_dims_vec, z_dims_vec);
    const T* dz_data = dz->data<T>();
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();

    if (dx != nullptr) {
      T* dx_data = dx->mutable_data<T>(ctx.GetPlace());
      if (rdims_for_x.size() == 0) {
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

        int ret = xpu::reduce_sum<XPUType>(
            dev_ctx.x_context(), reinterpret_cast<const XPUType*>(dz_data),
            reinterpret_cast<XPUType*>(dx_data), z_dims_vec, rdims_for_x);
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_sum ");
      }
    }

    if (dy != nullptr) {
      T* dy_data = dy->mutable_data<T>(ctx.GetPlace());
      if (rdims_for_y.size() == 0) {
        if (dy_data != dz_data) {
          framework::TensorCopy(
              *dz, ctx.GetPlace(),
              ctx.template device_context<platform::DeviceContext>(), dy);
        }
      } else {
        int ret = xpu::reduce_sum<XPUType>(
            dev_ctx.x_context(), reinterpret_cast<const XPUType*>(dz_data),
            reinterpret_cast<XPUType*>(dy_data), z_dims_vec, rdims_for_y);
        PADDLE_ENFORCE_XDNN_SUCCESS(ret, "reduce_sum ");
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
