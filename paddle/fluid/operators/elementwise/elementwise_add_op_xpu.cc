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
  void Compute(const framework::ExecutionContext &ctx) const override {
    XPUElementwise<T, XPUAddFunctor<T>>(ctx);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseAddGradXPUKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto dx_dims = dout->dims();
    auto dy_dims_untrimed = dout->dims();
    T *dx_data = NULL;
    T *dy_data = NULL;

    int axis = ctx.Attr<int>("axis");
    PADDLE_ENFORCE_GE(dx_dims.size(), dy_dims_untrimed.size(),
                      platform::errors::InvalidArgument(
                          "Rank of first input must >= rank of second input."));

    if (dx != nullptr) {
      dx->mutable_data<T>(ctx.GetPlace());
      dx_dims = dx->dims();
      dx_data = dx->data<T>();
    }

    if (dy != nullptr) {
      dy->mutable_data<T>(ctx.GetPlace());
      dy_dims_untrimed = dy->dims();
      dy_data = dy->data<T>();
    }

    int pre, n, post, is_common_broadcast;
    if (dx_dims == dy_dims_untrimed) {
      pre = post = 1;
      n = dout->numel();
    } else {
      axis = (axis == -1 ? dx_dims.size() - dy_dims_untrimed.size() : axis);
      PADDLE_ENFORCE_EQ(axis >= 0 && axis < dx_dims.size(), true,
                        platform::errors::InvalidArgument(
                            "Axis should be in range [0, dx_dims)"));
      auto dy_dims = trim_trailing_singular_dims(dy_dims_untrimed);
      axis = (dy_dims.size() == 0) ? dx_dims.size() : axis;
      get_mid_dims(dx_dims, dy_dims, axis, &pre, &n, &post,
                   &is_common_broadcast);
    }
    int len = pre * n * post;

    auto &dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    if (post == 1) {
      int r = xpu::matrix_vector_add_grad(
          dev_ctx.x_context(), dout->data<T>(), dout->data<T>(),
          dout->data<T>(), dout->data<T>(), dx_data, dy_data, pre, n);
      if (r == xpu::Error_t::INVALID_PARAM) {
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                          platform::errors::InvalidArgument(
                              "XPU kernel error of ElementWiseAddOp, error "
                              "message: INVALID_PARAM, "
                              "please check your input & output."));
      } else if (r == xpu::Error_t::RUNTIME_ERROR) {
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                          platform::errors::Unavailable(
                              "XPU kernel error of ElementWiseAddOp, error "
                              "message: RUNTIME_ERROR, "
                              "please check whether Baidu Kunlun card is "
                              "properly installed."));
      } else if (r == xpu::Error_t::NO_ENOUGH_WORKSPACE) {
        PADDLE_ENFORCE_EQ(
            r, xpu::Error_t::SUCCESS,
            platform::errors::ResourceExhausted(
                "XPU kernel error of ElementWiseAddOp, error message: "
                "NO_ENOUGH_WORKSPACE, XPU has no enough memory."));
      }
      return;
    }

    if (dx == nullptr) {
      PADDLE_ENFORCE_EQ(
          xpu_malloc(reinterpret_cast<void **>(&dx_data), len * sizeof(float)),
          XPU_SUCCESS,
          platform::errors::ResourceExhausted("XPU has no enough memory"));
    }

    if (dy == nullptr) {
      PADDLE_ENFORCE_EQ(
          xpu_malloc(reinterpret_cast<void **>(&dy_data), len * sizeof(float)),
          XPU_SUCCESS,
          platform::errors::ResourceExhausted("XPU has no enough memory"));
    } else {
      if (len != n) {
        PADDLE_ENFORCE_EQ(xpu_malloc(reinterpret_cast<void **>(&dy_data),
                                     len * sizeof(float)),
                          XPU_SUCCESS, platform::errors::ResourceExhausted(
                                           "XPU has no enough memory"));
      }
    }

    int r = xpu::elementwise_add_grad(
        dev_ctx.x_context(), dout->data<T>() /*x*/, dout->data<T>() /*y*/,
        dout->data<T>() /*out*/, dout->data<T>(), dx_data, dy_data, len);
    if (r == xpu::Error_t::INVALID_PARAM) {
      PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                        platform::errors::InvalidArgument(
                            "XPU kernel error of ElementWiseAddOp, error "
                            "message: INVALID_PARAM, "
                            "please check your input & output."));
    } else if (r == xpu::Error_t::RUNTIME_ERROR) {
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::Unavailable(
              "XPU kernel error of ElementWiseAddOp, error message: "
              "RUNTIME_ERROR, "
              "please check whether Baidu Kunlun card is properly installed."));
    } else if (r == xpu::Error_t::NO_ENOUGH_WORKSPACE) {
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::ResourceExhausted(
              "XPU kernel error of ElementWiseAddOp, error message: "
              "NO_ENOUGH_WORKSPACE, XPU has no enough memory."));
    }

    if ((dy != nullptr) && (len != n)) {
      r = xpu::reduce_ew(dev_ctx.x_context(), dy_data, dy->data<T>(), pre, n,
                         post, xpu::ElementwiseOp::ASSIGN);
      if (r == xpu::Error_t::INVALID_PARAM) {
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                          platform::errors::InvalidArgument(
                              "XPU kernel error of ElementWiseAddOp, error "
                              "message: INVALID_PARAM, "
                              "please check your input & output."));
      } else if (r == xpu::Error_t::RUNTIME_ERROR) {
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                          platform::errors::Unavailable(
                              "XPU kernel error of ElementWiseAddOp, error "
                              "message: RUNTIME_ERROR, "
                              "please check whether Baidu Kunlun card is "
                              "properly installed."));
      } else if (r == xpu::Error_t::NO_ENOUGH_WORKSPACE) {
        PADDLE_ENFORCE_EQ(
            r, xpu::Error_t::SUCCESS,
            platform::errors::ResourceExhausted(
                "XPU kernel error of ElementWiseAddOp, error message: "
                "NO_ENOUGH_WORKSPACE, XPU has no enough memory."));
      }
      dev_ctx.Wait();
      xpu_free(dy_data);
    }

    if ((dx == nullptr || dy == nullptr) && !(dy != nullptr && len != n)) {
      dev_ctx.Wait();
    }

    if (dx == nullptr) {
      xpu_free(dx_data);
    }
    if (dy == nullptr) {
      xpu_free(dy_data);
    }
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
