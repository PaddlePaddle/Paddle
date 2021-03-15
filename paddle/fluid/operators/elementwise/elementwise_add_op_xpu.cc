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

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ElementwiseAddXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // XPUElementwise<T>(ctx, xpu::add<T>);
    // ToDo(QingshuChen): update this optimization to elementwise_xpu.h
    auto x_var = ctx.InputVar("X");
    PADDLE_ENFORCE_NE(x_var, nullptr, platform::errors::InvalidArgument(
                                          "Cannot get input Variable X"));
    PADDLE_ENFORCE_EQ(
        x_var->IsType<framework::LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "XPU only support LoDTensor, Input(X) is not LoDTensor"));

    auto x = x_var->Get<framework::LoDTensor>();
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());
    auto x_dims = x.dims();
    auto y_dims = y->dims();
    int max_dim = std::max(x_dims.size(), y_dims.size());
    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

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
    const T* x_data = x.data<T>();
    const T* y_data = y->data<T>();
    T* z_data = z->data<T>();

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    int ret = xpu::SUCCESS;
    ret = xpu::broadcast_add<T>(dev_ctx.x_context(), x_data, y_data, z_data,
                                x_dims_vec, y_dims_vec);
    PADDLE_ENFORCE_EQ(
        ret, xpu::SUCCESS,
        platform::errors::External(
            "XPU kernel Elementwise occur error in XPUElementwise error code ",
            ret, XPUAPIErrorMsg[ret]));
  }
};

template <typename DeviceContext, typename T>
class ElementwiseAddGradXPUKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    // XPUElementwiseGrad<T>(ctx, xpu::add_grad<T>, false);
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dz = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    const framework::DDim& x_dims = x->dims();
    const framework::DDim& y_dims = y->dims();
    int max_dim = std::max(x_dims.size(), y_dims.size());
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
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

    T* dx_data = nullptr;
    T* dy_data = nullptr;
    if (dx) {
      dx_data = dx->mutable_data<T>(ctx.GetPlace());
    }
    if (dy) {
      dy_data = dy->mutable_data<T>(ctx.GetPlace());
    }

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    int ret = xpu::broadcast_add_grad<T>(dev_ctx.x_context(), dx_data, dx_data,
                                         dx_data, dz->data<T>(), dy_data,
                                         dx_data, x_dims_vec, y_dims_vec);
    PADDLE_ENFORCE_EQ(
        ret, xpu::SUCCESS,
        platform::errors::External(
            "XPU kernel Elementwise occur error in XPUElementwise error code ",
            ret, XPUAPIErrorMsg[ret]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    elementwise_add,
    ops::ElementwiseAddXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(elementwise_add_grad,
                       ops::ElementwiseAddGradXPUKernel<
                           paddle::platform::XPUDeviceContext, float>);
#endif
