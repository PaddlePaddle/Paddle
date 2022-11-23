/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AffineChannelXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");

    auto* y = ctx.Output<phi::DenseTensor>("Out");
    y->mutable_data<T>(ctx.GetPlace());

    const framework::DataLayout layout =
        framework::StringToDataLayout(ctx.Attr<std::string>("data_layout"));

    auto dims = x->dims();
    int N = dims[0];
    int C = layout == framework::DataLayout::kNCHW ? dims[1]
                                                   : dims[dims.size() - 1];
    int HxW = x->numel() / N / C;

    auto* scale_d = scale->data<T>();
    auto* bias_d = bias->data<T>();

    auto* x_d = x->data<T>();
    auto* y_d = y->data<T>();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    std::vector<int> x_shape;
    std::vector<int> b_shape;
    if (layout == framework::DataLayout::kNCHW) {
      x_shape.push_back(N);
      x_shape.push_back(C);
      x_shape.push_back(HxW);
      b_shape.push_back(1);
      b_shape.push_back(C);
      b_shape.push_back(1);
    } else {
      x_shape.push_back(N * HxW);
      x_shape.push_back(C);
      b_shape.push_back(1);
      b_shape.push_back(C);
    }
    int r = 0;
    r = xpu::broadcast_mul(
        dev_ctx.x_context(), x_d, scale_d, y_d, x_shape, b_shape);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "The broadcast_mul XPU OP return wrong value[%d %s]",
                          r,
                          XPUAPIErrorMsg[r]));
    r = xpu::broadcast_add(
        dev_ctx.x_context(), y_d, bias_d, y_d, x_shape, b_shape);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "The broadcast_add XPU OP return wrong value[%d %s]",
                          r,
                          XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class AffineChannelGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* dy = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dscale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    const framework::DataLayout layout =
        framework::StringToDataLayout(ctx.Attr<std::string>("data_layout"));

    auto dims = x->dims();
    int N = dims[0];
    int C = layout == framework::DataLayout::kNCHW ? dims[1]
                                                   : dims[dims.size() - 1];
    int HxW = x->numel() / N / C;

    auto* dy_d = dy->data<T>();
    auto* scale_d = scale->data<T>();

    T* dx_d = dx ? dx->mutable_data<T>(ctx.GetPlace()) : nullptr;
    T* dscale_d = dscale ? dscale->mutable_data<T>(ctx.GetPlace()) : nullptr;
    T* dbias_d = dbias ? dbias->mutable_data<T>(ctx.GetPlace()) : nullptr;

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    std::vector<int> x_shape;
    std::vector<int> b_shape;
    std::vector<int> rdims;
    if (layout == framework::DataLayout::kNCHW) {
      x_shape.push_back(N);
      x_shape.push_back(C);
      x_shape.push_back(HxW);
      b_shape.push_back(1);
      b_shape.push_back(C);
      b_shape.push_back(1);
      rdims.push_back(0);
      rdims.push_back(2);
    } else {
      x_shape.push_back(N * HxW);
      x_shape.push_back(C);
      b_shape.push_back(1);
      b_shape.push_back(C);
      rdims.push_back(0);
    }

    int r = 0;
    if (dscale_d && dbias_d) {
      r = xpu::reduce_sum<T>(
          dev_ctx.x_context(), dy_d, dbias_d, x_shape, rdims);
      PADDLE_ENFORCE_EQ(r,
                        xpu::Error_t::SUCCESS,
                        platform::errors::External(
                            "The reduce_sum XPU OP return wrong value[%d %s]",
                            r,
                            XPUAPIErrorMsg[r]));
      T* tmp = nullptr;
      r = xpu_malloc(reinterpret_cast<void**>(&tmp), dy->numel() * sizeof(T));
      PADDLE_ENFORCE_EQ(r,
                        xpu::Error_t::SUCCESS,
                        platform::errors::External("no enough memory in xpu"));

      r = xpu::mul<T>(
          dev_ctx.x_context(), dy_d, x->data<T>(), tmp, dy->numel());
      PADDLE_ENFORCE_EQ(
          r,
          xpu::Error_t::SUCCESS,
          platform::errors::External("The mul XPU OP return wrong value[%d %s]",
                                     r,
                                     XPUAPIErrorMsg[r]));
      r = xpu::reduce_sum<T>(
          dev_ctx.x_context(), tmp, dscale_d, x_shape, rdims);
      PADDLE_ENFORCE_EQ(r,
                        xpu::Error_t::SUCCESS,
                        platform::errors::External(
                            "The reduce_sum XPU OP return wrong value[%d %s]",
                            r,
                            XPUAPIErrorMsg[r]));
      if (dev_ctx.x_context()->xpu_stream) {
        dev_ctx.Wait();
      }
      xpu_free(tmp);
    }
    if (dx_d) {
      r = xpu::broadcast_mul(
          dev_ctx.x_context(), dy_d, scale_d, dx_d, x_shape, b_shape);
      PADDLE_ENFORCE_EQ(
          r,
          xpu::Error_t::SUCCESS,
          platform::errors::External(
              "The broadcast_mul XPU OP return wrong value[%d %s]",
              r,
              XPUAPIErrorMsg[r]));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using XPU = paddle::platform::XPUDeviceContext;

REGISTER_OP_XPU_KERNEL(affine_channel, ops::AffineChannelXPUKernel<XPU, float>);
REGISTER_OP_XPU_KERNEL(affine_channel_grad,
                       ops::AffineChannelGradXPUKernel<XPU, float>);

#endif
