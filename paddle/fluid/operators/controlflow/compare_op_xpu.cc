/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

template <typename T, typename XPUType>
void XPUCompare(
    const framework::ExecutionContext& ctx,
    std::function<int(xpu::Context*, const XPUType*, const XPUType*, bool*,
                      const std::vector<int>&, const std::vector<int>&)>
        func) {
  auto* x = ctx.Input<framework::Tensor>("X");
  auto* y = ctx.Input<framework::Tensor>("Y");
  auto* z = ctx.Output<framework::Tensor>("Out");

  auto x_shape = phi::vectorize<int>(x->dims());
  auto y_shape = phi::vectorize<int>(y->dims());

  auto x_data = reinterpret_cast<const XPUType*>(x->data<T>());
  auto y_data = reinterpret_cast<const XPUType*>(y->data<T>());
  auto z_data = z->mutable_data<bool>(ctx.GetPlace());

  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();

  int ret = func(dev_ctx.x_context(), x_data, y_data, z_data, x_shape, y_shape);
  PADDLE_ENFORCE_EQ(
      ret, xpu::SUCCESS,
      platform::errors::External(
          "XPU kernel compare op occur error[%d %s] in XPUCompare.", ret,
          XPUAPIErrorMsg[ret]));
}

template <typename DeviceContext, typename T>
class EqualXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    XPUCompare<T, XPUType>(ctx, xpu::broadcast_equal<XPUType>);
  }
};

template <typename DeviceContext, typename T>
class NotEqualXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    XPUCompare<T, XPUType>(ctx, xpu::broadcast_not_equal<XPUType>);
  }
};

template <typename DeviceContext, typename T>
class LessThanXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    XPUCompare<T, XPUType>(ctx, xpu::broadcast_less_than<XPUType>);
  }
};

template <typename DeviceContext, typename T>
class LessEqualXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    XPUCompare<T, XPUType>(ctx, xpu::broadcast_less_equal<XPUType>);
  }
};

template <typename DeviceContext, typename T>
class GreaterThanXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    XPUCompare<T, XPUType>(ctx, xpu::broadcast_greater_than<XPUType>);
  }
};

template <typename DeviceContext, typename T>
class GreaterEqualXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    XPUCompare<T, XPUType>(ctx, xpu::broadcast_greater_equal<XPUType>);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(equal,
                       ops::EqualXPUKernel<plat::XPUDeviceContext, float>,
                       ops::EqualXPUKernel<plat::XPUDeviceContext, int>,
                       ops::EqualXPUKernel<plat::XPUDeviceContext, int64_t>);

REGISTER_OP_XPU_KERNEL(not_equal,
                       ops::NotEqualXPUKernel<plat::XPUDeviceContext, float>,
                       ops::NotEqualXPUKernel<plat::XPUDeviceContext, int>,
                       ops::NotEqualXPUKernel<plat::XPUDeviceContext, int64_t>);

REGISTER_OP_XPU_KERNEL(less_than,
                       ops::LessThanXPUKernel<plat::XPUDeviceContext, float>,
                       ops::LessThanXPUKernel<plat::XPUDeviceContext, int>,
                       ops::LessThanXPUKernel<plat::XPUDeviceContext, int64_t>);

REGISTER_OP_XPU_KERNEL(
    less_equal, ops::LessEqualXPUKernel<plat::XPUDeviceContext, float>,
    ops::LessEqualXPUKernel<plat::XPUDeviceContext, int>,
    ops::LessEqualXPUKernel<plat::XPUDeviceContext, int64_t>);

REGISTER_OP_XPU_KERNEL(
    greater_than, ops::GreaterThanXPUKernel<plat::XPUDeviceContext, float>,
    ops::GreaterThanXPUKernel<plat::XPUDeviceContext, int>,
    ops::GreaterThanXPUKernel<plat::XPUDeviceContext, int64_t>);

REGISTER_OP_XPU_KERNEL(
    greater_equal, ops::GreaterEqualXPUKernel<plat::XPUDeviceContext, float>,
    ops::GreaterEqualXPUKernel<plat::XPUDeviceContext, int>,
    ops::GreaterEqualXPUKernel<plat::XPUDeviceContext, int64_t>);

#endif
