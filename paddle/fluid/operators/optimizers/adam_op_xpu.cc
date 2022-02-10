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

#include "paddle/fluid/operators/optimizers/adam_op.h"
#include "gflags/gflags.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#ifdef PADDLE_WITH_XPU
template <typename DeviceContext, typename T>
class AdamOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong type，Expected Var(%s)'s "
                          "type is LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));
    using paddle::framework::LoDTensor;

    auto& param = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Param"), "Input",
                                  "Param", "Adam");
    // auto& grad = Ref(ctx.Input<LoDTensor>("Grad"), "Must set Grad");
    auto* grad_var = ctx.InputVar("Grad");
    auto& mom1 = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Moment1"), "Input",
                                 "Moment1", "Adam");
    auto& mom2 = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Moment2"), "Input",
                                 "Moment2", "Adam");
    auto& lr = GET_DATA_SAFELY(ctx.Input<LoDTensor>("LearningRate"), "Input",
                               "LearningRate", "Adam");
    auto& beta1_pow = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Beta1Pow"), "Input",
                                      "Beta1Pow", "Adam");
    auto& beta2_pow = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Beta2Pow"), "Input",
                                      "Beta2Pow", "Adam");

    auto& param_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("ParamOut"),
                                      "Output", "ParamOut", "Adam");
    auto& mom1_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Moment1Out"),
                                     "Output", "Moment1Out", "Adam");
    auto& mom2_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("Moment2Out"),
                                     "Output", "Moment2Out", "Adam");

    auto* beta1_pow_out = ctx.Output<LoDTensor>("Beta1PowOut");
    auto* beta2_pow_out = ctx.Output<LoDTensor>("Beta2PowOut");

    bool skip_update = false;
    if (ctx.HasInput("SkipUpdate")) {
      auto* skip_update_tensor = ctx.Input<framework::Tensor>("SkipUpdate");
      PADDLE_ENFORCE_EQ(skip_update_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(SkipUpdate) size must be 1, but get %d",
                            skip_update_tensor->numel()));
      std::vector<bool> skip_update_vec;
      paddle::framework::TensorToVector(*skip_update_tensor,
                                        ctx.device_context(), &skip_update_vec);
      skip_update = skip_update_vec[0];
    }
    // skip_update=true, just copy input to output, and TensorCopy will call
    // mutable_data
    if (skip_update) {
      VLOG(4) << "Adam skip update";
      framework::TensorCopy(
          param, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), &param_out);
      framework::TensorCopy(
          mom1, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), &mom1_out);
      framework::TensorCopy(
          mom2, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), &mom2_out);
      framework::TensorCopy(
          beta1_pow, beta1_pow.place(),
          ctx.template device_context<platform::DeviceContext>(),
          beta1_pow_out);
      framework::TensorCopy(
          beta2_pow, beta2_pow.place(),
          ctx.template device_context<platform::DeviceContext>(),
          beta2_pow_out);
      return;
    }

    PADDLE_ENFORCE_EQ(beta1_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong size, Expected beta1 pow "
                          "output size is 1, but received "
                          "value is:%d.",
                          beta1_pow_out->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong size, Expected beta2 pow "
                          "output size is 1, but received "
                          "value is:%d.",
                          beta2_pow_out->numel()));

    bool use_global_beta_pow = ctx.Attr<bool>("use_global_beta_pow");
    VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

    float beta1 = static_cast<float>(ctx.Attr<float>("beta1"));
    if (ctx.HasInput("Beta1Tensor")) {
      auto* beta1_tensor = ctx.Input<framework::Tensor>("Beta1Tensor");
      beta1 = static_cast<float>(GetAttrFromTensor(beta1_tensor));
    }
    float beta2 = static_cast<float>(ctx.Attr<float>("beta2"));
    if (ctx.HasInput("Beta2Tensor")) {
      auto* beta2_tensor = ctx.Input<framework::Tensor>("Beta2Tensor");
      beta2 = static_cast<float>(GetAttrFromTensor(beta2_tensor));
    }
    float epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    if (ctx.HasInput("EpsilonTensor")) {
      auto* epsilon_tensor = ctx.Input<framework::Tensor>("EpsilonTensor");
      epsilon = static_cast<float>(GetAttrFromTensor(epsilon_tensor));
    }
    if (grad_var->IsType<framework::LoDTensor>()) {
      auto& grad = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Grad"), "Input",
                                   "Grad", "Adam");
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      const float* beta1_pow_ptr = beta1_pow.template data<float>();
      const float* beta2_pow_ptr = beta2_pow.template data<float>();
      Tensor xpu_beta1_pow;
      Tensor xpu_beta2_pow;
      if (beta1_pow.place() == platform::CPUPlace() &&
          beta2_pow.place() == platform::CPUPlace()) {
        paddle::framework::TensorCopy(beta1_pow, ctx.GetPlace(), dev_ctx,
                                      &xpu_beta1_pow);
        paddle::framework::TensorCopy(beta2_pow, ctx.GetPlace(), dev_ctx,
                                      &xpu_beta2_pow);
        dev_ctx.Wait();
        beta1_pow_ptr = xpu_beta1_pow.template data<float>();
        beta2_pow_ptr = xpu_beta2_pow.template data<float>();
      }

      int r = xpu::adam(dev_ctx.x_context(), grad.template data<T>(),
                        mom1.template data<T>(), mom2.template data<T>(),
                        param.template data<float>(), beta1_pow_ptr,
                        beta2_pow_ptr, lr.template data<float>(),
                        mom1_out.template mutable_data<float>(ctx.GetPlace()),
                        mom2_out.template mutable_data<float>(ctx.GetPlace()),
                        param_out.template mutable_data<float>(ctx.GetPlace()),
                        beta1, beta2, epsilon, param.numel());

      xpu_wait(dev_ctx.x_context()->xpu_stream);
      PADDLE_ENFORCE_EQ(
          r == xpu::Error_t::SUCCESS, true,
          platform::errors::External("XPU API return wrong value[%d],", r));
      if (!use_global_beta_pow) {
        // update in cpu and then copy to xpu
        if (beta1_pow.place() == platform::CPUPlace() &&
            beta2_pow.place() == platform::CPUPlace()) {
          const float* beta1_pow_p = beta1_pow.template data<float>();
          beta1_pow_out->mutable_data<float>(platform::CPUPlace())[0] =
              beta1 * beta1_pow_p[0];
          const float* beta2_pow_p = beta2_pow.template data<float>();
          beta2_pow_out->mutable_data<float>(platform::CPUPlace())[0] =
              beta2 * beta2_pow_p[0];
        } else {
          float* beta1_pow_out_p =
              beta1_pow_out->mutable_data<float>(ctx.GetPlace());
          float* beta2_pow_out_p =
              beta2_pow_out->mutable_data<float>(ctx.GetPlace());
          int r =
              xpu::scale(dev_ctx.x_context(), beta1_pow_ptr, beta1_pow_out_p,
                         beta1_pow.numel(), false, beta1, 0.0f);
          PADDLE_ENFORCE_EQ(
              r, xpu::SUCCESS,
              platform::errors::External(
                  "XPU kernel scale occur error in adam error code ", r,
                  XPUAPIErrorMsg[r]));
          r = xpu::scale(dev_ctx.x_context(), beta2_pow_ptr, beta2_pow_out_p,
                         beta2_pow.numel(), false, beta2, 0.0f);
          PADDLE_ENFORCE_EQ(
              r, xpu::SUCCESS,
              platform::errors::External(
                  "XPU kernel scale occur error in adam error code ", r,
                  XPUAPIErrorMsg[r]));

          xpu_wait(dev_ctx.x_context()->xpu_stream);
        }
      }
    } else if (grad_var->IsType<pten::SelectedRows>()) {
      auto* grad = ctx.Input<pten::SelectedRows>("Grad");
      auto& dev_ctx = ctx.template device_context<DeviceContext>();

      if (grad->rows().size() == 0) {
        VLOG(3) << "grad row size is 0!!";
        return;
      }

      std::vector<int64_t> cpu_rows(grad->rows().begin(), grad->rows().end());
      bool is_strict_sorted = true;
      for (size_t i = 1; i < cpu_rows.size(); ++i) {
        if (cpu_rows[i - 1] >= cpu_rows[i]) {
          is_strict_sorted = false;
          break;
        }
      }

      pten::SelectedRows tmp_grad_merge;
      const pten::SelectedRows* grad_merge_ptr;
      if (is_strict_sorted) {
        grad_merge_ptr = grad;
      } else {
        scatter::MergeAdd<platform::XPUDeviceContext, T> merge_func;
        merge_func(ctx.template device_context<platform::XPUDeviceContext>(),
                   *grad, &tmp_grad_merge, true);

        xpu_wait(dev_ctx.x_context()->xpu_stream);
        grad_merge_ptr = &tmp_grad_merge;
      }
      const T* beta1_pow_ptr = beta1_pow.template data<T>();
      const T* beta2_pow_ptr = beta2_pow.template data<T>();
      Tensor xpu_beta1_pow;
      Tensor xpu_beta2_pow;
      if (beta1_pow.place() == platform::CPUPlace() &&
          beta2_pow.place() == platform::CPUPlace()) {
        paddle::framework::TensorCopy(beta1_pow, ctx.GetPlace(), dev_ctx,
                                      &xpu_beta1_pow);
        paddle::framework::TensorCopy(beta2_pow, ctx.GetPlace(), dev_ctx,
                                      &xpu_beta2_pow);
        dev_ctx.Wait();
        beta1_pow_ptr = xpu_beta1_pow.template data<T>();
        beta2_pow_ptr = xpu_beta2_pow.template data<T>();
      }
      auto& grad_merge = *grad_merge_ptr;
      auto& grad_tensor = grad_merge.value();
      const T* grad_data = grad_tensor.template data<T>();
      int row_count = grad_merge.rows().size();
      std::vector<int> rows(row_count);
      xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
      int* xpu_rows = RAII_GUARD.alloc_l3_or_gm<int>(row_count);
      std::vector<int64_t> merge_rows(grad_merge.rows().begin(),
                                      grad_merge.rows().end());
      for (size_t i = 0; i < grad_merge.rows().size(); ++i) {
        rows[i] = static_cast<int>(merge_rows[i]);
      }
      xpu_wait(dev_ctx.x_context()->xpu_stream);
      memory::Copy(ctx.GetPlace(), xpu_rows, platform::CPUPlace(), rows.data(),
                   row_count * sizeof(int));
      auto row_numel = grad_tensor.numel() / grad_merge.rows().size();
      auto ori_rows = param.numel() / row_numel;

      int lazy_mode = static_cast<int>(ctx.Attr<bool>("lazy_mode"));
      int r = xpu::sparse_adam(
          dev_ctx.x_context(), grad_data, mom1.template data<T>(),
          mom2.template data<T>(), param.template data<T>(), beta1_pow_ptr,
          beta2_pow_ptr, lr.template data<T>(),
          mom1_out.template mutable_data<T>(ctx.GetPlace()),
          mom2_out.template mutable_data<T>(ctx.GetPlace()),
          param_out.template mutable_data<T>(ctx.GetPlace()), beta1, beta2,
          epsilon, ori_rows, xpu_rows, row_numel, grad_merge.rows().size(),
          lazy_mode);

      PADDLE_ENFORCE_EQ(
          r == xpu::Error_t::SUCCESS, true,
          platform::errors::External("XPU API return wrong value[%d],", r));

      if (!use_global_beta_pow) {
        // update in cpu and then copy to xpu
        if (beta1_pow.place() == platform::CPUPlace() &&
            beta2_pow.place() == platform::CPUPlace()) {
          const float* beta1_pow_p = beta1_pow.template data<float>();
          beta1_pow_out->mutable_data<float>(platform::CPUPlace())[0] =
              beta1 * beta1_pow_p[0];
          const float* beta2_pow_p = beta2_pow.template data<float>();
          beta2_pow_out->mutable_data<float>(platform::CPUPlace())[0] =
              beta2 * beta2_pow_p[0];
        } else {
          float* beta1_pow_out_p =
              beta1_pow_out->mutable_data<float>(ctx.GetPlace());
          float* beta2_pow_out_p =
              beta2_pow_out->mutable_data<float>(ctx.GetPlace());
          int r =
              xpu::scale(dev_ctx.x_context(), beta1_pow_ptr, beta1_pow_out_p,
                         beta1_pow.numel(), false, beta1, 0.0f);
          PADDLE_ENFORCE_EQ(
              r, xpu::SUCCESS,
              platform::errors::External(
                  "XPU kernel scale occur error in adam error code ", r,
                  XPUAPIErrorMsg[r]));
          r = xpu::scale(dev_ctx.x_context(), beta2_pow_ptr, beta2_pow_out_p,
                         beta2_pow.numel(), false, beta2, 0.0f);
          PADDLE_ENFORCE_EQ(
              r, xpu::SUCCESS,
              platform::errors::External(
                  "XPU kernel scale occur error in adam error code ", r,
                  XPUAPIErrorMsg[r]));
        }
      }
      xpu_wait(dev_ctx.x_context()->xpu_stream);
    } else {
      PADDLE_ENFORCE_EQ(1, 2, platform::errors::InvalidArgument(
                                  "Variable type not supported by adam_op"));
    }
  }
};
#endif

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
#ifdef PADDLE_WITH_XPU
REGISTER_OP_XPU_KERNEL(
    adam, ops::AdamOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
