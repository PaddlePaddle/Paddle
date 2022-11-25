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
#include <string>

#include "paddle/fluid/operators/optimizers/sgd_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SGDOpXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    const auto *param_var = ctx.InputVar("Param");
    const auto *grad_var = ctx.InputVar("Grad");

    if (param_var->IsType<framework::LoDTensor>() &&
        grad_var->IsType<framework::LoDTensor>()) {
      const auto *param = ctx.Input<framework::Tensor>("Param");
      auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
      // Actually, all tensors are LoDTensor except SelectedRows.
      const auto *grad = ctx.Input<framework::Tensor>("Grad");
      auto sz = param_out->numel();
      PADDLE_ENFORCE_EQ(param->numel(),
                        sz,
                        platform::errors::InvalidArgument(
                            "The input tensor Param's numel of SgdOp "
                            "should be equal with ParamOut's numel. "
                            "But received Param's "
                            "numel = [%s], ParamOut's numel = [%s]",
                            param->numel(),
                            sz));
      PADDLE_ENFORCE_EQ(grad->numel(),
                        sz,
                        platform::errors::InvalidArgument(
                            "The input tensor Grad's numel of SgdOp "
                            "should be equal with ParamOut's numel. "
                            "But received Grad's "
                            "numel = [%s], ParamOut's numel = [%s]",
                            grad->numel(),
                            sz));

      const T *lr_t = learning_rate->data<T>();
      auto &dev_ctx = ctx.template device_context<DeviceContext>();
      xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
      const float *lr = nullptr;
      if (std::is_same<T, paddle::platform::float16>::value) {
        float *lr_float =
            RAII_GUARD.alloc_l3_or_gm<float>(learning_rate->numel());
        int r = xpu::cast_v2<XPUType, float>(
            dev_ctx.x_context(),
            reinterpret_cast<const XPUType *>(lr_t),
            lr_float,
            learning_rate->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip_v2");
        lr = lr_float;
      } else {
        lr = reinterpret_cast<const float *>(lr_t);
      }

      const T *param_data = param->data<T>();
      const T *grad_data = grad->data<T>();
      T *out_data = param_out->mutable_data<T>(ctx.GetPlace());

      int r = xpu::sgd(dev_ctx.x_context(),
                       reinterpret_cast<const XPUType *>(grad_data),
                       reinterpret_cast<const XPUType *>(param_data),
                       lr,
                       reinterpret_cast<XPUType *>(out_data),
                       sz);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "sgd");
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(
    sgd,
    ops::SGDOpXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::SGDOpXPUKernel<paddle::platform::XPUDeviceContext, plat::float16>);
#endif
