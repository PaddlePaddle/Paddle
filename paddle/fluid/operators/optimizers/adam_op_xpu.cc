/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
THOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gflags/gflags.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/optimizers/adam_op_functor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using float16 = paddle::platform::float16;

#ifdef PADDLE_WITH_XPU
template <typename T1, typename T2>
static int ConvertDataByType(T1* x,
                             T2** y,
                             int len,
                             bool allocateFlag,
                             const framework::ExecutionContext& ctx) {
  if (nullptr == x || nullptr == y || len <= 0)
    return xpu::Error_t::INVALID_PARAM;
  int r = 0;
  if (allocateFlag) {
    r = xpu_malloc(reinterpret_cast<void**>(y), sizeof(T2) * len);

    PADDLE_ENFORCE_EQ(r,
                      0,
                      platform::errors::External(
                          "Alloc memory in xpu for result data failed"));
  }

  T1* cpu_data = reinterpret_cast<T1*>(malloc(sizeof(T1) * len));

  paddle::memory::Copy(paddle::platform::CPUPlace(),
                       cpu_data,
                       ctx.GetPlace(),
                       x,
                       len * sizeof(T1));

  T2* cpu_real_data = reinterpret_cast<T2*>(malloc(sizeof(T2) * len));
  for (int i = 0; i < len; i++) cpu_real_data[i] = static_cast<T2>(cpu_data[i]);

  paddle::memory::Copy(ctx.GetPlace(),
                       *y,
                       paddle::platform::CPUPlace(),
                       cpu_real_data,
                       len * sizeof(T2));

  free(cpu_data);
  free(cpu_real_data);

  return xpu::Error_t::SUCCESS;
}

template <typename T>
static void getDataPointer(const phi::DenseTensor& tensorData,
                           T** result,
                           const framework::ExecutionContext& ctx) {
  if (tensorData.dtype() == paddle::experimental::DataType::FLOAT16) {
    float16* real_data = const_cast<float16*>(
        tensorData.template data<paddle::platform::float16>());
    int len = tensorData.numel();

    int r = ConvertDataByType<float16, T>(real_data, result, len, true, ctx);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "execute function ConvertDataByType failed"));
  } else {
    *result = const_cast<T*>(tensorData.template data<T>());
  }
}

template <typename T>
static void getOutDataPointer(phi::DenseTensor* tensorData,
                              Tensor* out,
                              T** result,
                              const framework::ExecutionContext& ctx) {
  if (tensorData->dtype() == paddle::experimental::DataType::FLOAT16) {
    *result = out->template mutable_data<T>(ctx.GetPlace());
  } else {
    *result = tensorData->template mutable_data<T>(ctx.GetPlace());
  }
}

template <typename T>
static void copyOutData(const Tensor& srcTensor,
                        phi::DenseTensor* dstTensor,
                        const framework::ExecutionContext& ctx) {
  if (dstTensor->dtype() == paddle::experimental::DataType::FLOAT16) {
    T* xpu_out_data = const_cast<T*>(srcTensor.template data<T>());
    float16* out_data =
        dstTensor->template mutable_data<float16>(ctx.GetPlace());

    int len = srcTensor.numel();

    int r =
        ConvertDataByType<T, float16>(xpu_out_data, &out_data, len, false, ctx);
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "execute function ConvertDataByType failed"));
  }
}

template <typename T>
static void setBetaData(const phi::DenseTensor& beta_pow,
                        phi::DenseTensor* beta_pow_out,
                        const T& beta) {
  if (beta_pow.dtype() == paddle::experimental::DataType::FLOAT16) {
    const float16* beta_pow_p = beta_pow.template data<float16>();
    beta_pow_out->mutable_data<float16>(platform::CPUPlace())[0] =
        static_cast<float16>(beta) * beta_pow_p[0];
  } else {
    const T* beta_pow_p = beta_pow.template data<T>();
    beta_pow_out->mutable_data<T>(platform::CPUPlace())[0] =
        beta * beta_pow_p[0];
  }
}

template <typename DeviceContext, typename T>
static void scale(phi::DenseTensor* beta_pow_out,
                  const phi::DenseTensor& beta_pow,
                  T* beta_pow_ptr,
                  const T& beta,
                  const framework::ExecutionContext& ctx) {
  float16* beta_pow_out_p2 =
      beta_pow_out->mutable_data<float16>(ctx.GetPlace());

  Tensor xpu_beta_pow_out;
  const phi::DenseTensorMeta meta_beta_pow_out(
      paddle::experimental::DataType::FLOAT32, beta_pow_out->dims());
  xpu_beta_pow_out.set_meta(meta_beta_pow_out);

  T* beta_pow_out_ptr =
      xpu_beta_pow_out.template mutable_data<T>(ctx.GetPlace());

  auto& dev_ctx = ctx.template device_context<DeviceContext>();
  int r = xpu::scale(dev_ctx.x_context(),
                     beta_pow_ptr,
                     beta_pow_out_ptr,
                     beta_pow.numel(),
                     false,
                     beta,
                     0.0f);

  xpu_wait(dev_ctx.x_context()->xpu_stream);
  PADDLE_ENFORCE_EQ(r,
                    xpu::SUCCESS,
                    platform::errors::External(
                        "XPU kernel scale occur error in adam error code ",
                        r,
                        XPUAPIErrorMsg[r]));

  float* xpu_beta_pow_out_data =
      const_cast<T*>(xpu_beta_pow_out.template data<T>());
  int len = xpu_beta_pow_out.numel();

  r = ConvertDataByType<T, float16>(
      xpu_beta_pow_out_data, &beta_pow_out_p2, len, false, ctx);
  PADDLE_ENFORCE_EQ(
      r,
      xpu::Error_t::SUCCESS,
      platform::errors::External("execute function ConvertDataByType failed"));
}

template <typename T>
static void freeData(const phi::DenseTensor& tensorData, T* dataPtr) {
  if (tensorData.dtype() == paddle::experimental::DataType::FLOAT16)
    xpu_free(dataPtr);
}

template <typename DeviceContext, typename T>
class AdamOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(),
                      true,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong type，Expected Var(%s)'s "
                          "type is LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));
    using paddle::framework::LoDTensor;

    auto& param = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>("Param"), "Input", "Param", "Adam");

    float* param_ptr = nullptr;
    getDataPointer<float>(param, &param_ptr, ctx);

    auto* grad_var = ctx.InputVar("Grad");
    float* grad_c = nullptr;

    auto& mom1 = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>("Moment1"), "Input", "Moment1", "Adam");
    float* mom1_ptr = nullptr;
    getDataPointer<float>(mom1, &mom1_ptr, ctx);

    auto& mom2 = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>("Moment2"), "Input", "Moment2", "Adam");
    float* mom2_ptr = nullptr;
    getDataPointer<float>(mom2, &mom2_ptr, ctx);

    auto& lr = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>("LearningRate"), "Input", "LearningRate", "Adam");
    float* lr_ptr = nullptr;
    getDataPointer<float>(lr, &lr_ptr, ctx);

    auto& beta1_pow = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>("Beta1Pow"), "Input", "Beta1Pow", "Adam");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    float* beta1_pow_ptr = nullptr;
    if (beta1_pow.place() == platform::CPUPlace()) {
      Tensor xpu_beta1_pow;
      paddle::framework::TensorCopy(
          beta1_pow, ctx.GetPlace(), dev_ctx, &xpu_beta1_pow);
      if (xpu_beta1_pow.dtype() == paddle::experimental::DataType::FLOAT16)
        getDataPointer<float>(xpu_beta1_pow, &beta1_pow_ptr, ctx);
      else
        beta1_pow_ptr =
            const_cast<float*>(xpu_beta1_pow.template data<float>());
    } else {
      if (beta1_pow.dtype() == paddle::experimental::DataType::FLOAT16)
        getDataPointer<float>(beta1_pow, &beta1_pow_ptr, ctx);
      else
        beta1_pow_ptr = const_cast<float*>(beta1_pow.template data<float>());
    }

    auto& beta2_pow = GET_DATA_SAFELY(
        ctx.Input<LoDTensor>("Beta2Pow"), "Input", "Beta2Pow", "Adam");
    float* beta2_pow_ptr = nullptr;
    if (beta2_pow.place() == platform::CPUPlace()) {
      Tensor xpu_beta2_pow;
      paddle::framework::TensorCopy(
          beta2_pow, ctx.GetPlace(), dev_ctx, &xpu_beta2_pow);
      if (xpu_beta2_pow.dtype() == paddle::experimental::DataType::FLOAT16)
        getDataPointer<float>(xpu_beta2_pow, &beta2_pow_ptr, ctx);
      else
        beta2_pow_ptr =
            const_cast<float*>(xpu_beta2_pow.template data<float>());
    } else {
      if (beta2_pow.dtype() == paddle::experimental::DataType::FLOAT16)
        getDataPointer<float>(beta2_pow, &beta2_pow_ptr, ctx);
      else
        beta2_pow_ptr = const_cast<float*>(beta2_pow.template data<float>());
    }

    auto& param_out = GET_DATA_SAFELY(
        ctx.Output<LoDTensor>("ParamOut"), "Output", "ParamOut", "Adam");
    Tensor xpu_param_out;
    float* param_out_ptr = nullptr;
    const phi::DenseTensorMeta meta_param(
        paddle::experimental::DataType::FLOAT32, param_out.dims());
    xpu_param_out.set_meta(meta_param);
    getOutDataPointer(&param_out, &xpu_param_out, &param_out_ptr, ctx);

    auto& mom1_out = GET_DATA_SAFELY(
        ctx.Output<LoDTensor>("Moment1Out"), "Output", "Moment1Out", "Adam");
    Tensor xpu_mom1_out;
    float* mom1_out_ptr = nullptr;
    const phi::DenseTensorMeta meta_mom1(
        paddle::experimental::DataType::FLOAT32, mom1_out.dims());
    xpu_mom1_out.set_meta(meta_mom1);
    getOutDataPointer(&mom1_out, &xpu_mom1_out, &mom1_out_ptr, ctx);

    auto& mom2_out = GET_DATA_SAFELY(
        ctx.Output<LoDTensor>("Moment2Out"), "Output", "Moment2Out", "Adam");
    Tensor xpu_mom2_out;
    float* mom2_out_ptr = nullptr;
    const phi::DenseTensorMeta meta_mom2(
        paddle::experimental::DataType::FLOAT32, mom2_out.dims());
    xpu_mom2_out.set_meta(meta_mom2);
    getOutDataPointer(&mom2_out, &xpu_mom2_out, &mom2_out_ptr, ctx);

    auto* beta1_pow_out = ctx.Output<LoDTensor>("Beta1PowOut");
    auto* beta2_pow_out = ctx.Output<LoDTensor>("Beta2PowOut");

    bool skip_update = false;
    if (ctx.HasInput("SkipUpdate")) {
      auto* skip_update_tensor = ctx.Input<framework::Tensor>("SkipUpdate");
      PADDLE_ENFORCE_EQ(skip_update_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(SkipUpdate) size must be 1, but get %d",
                            skip_update_tensor->numel()));
      std::vector<bool> skip_update_vec;
      paddle::framework::TensorToVector(
          *skip_update_tensor, ctx.device_context(), &skip_update_vec);
      skip_update = skip_update_vec[0];
    }
    // skip_update=true, just copy input to output, and TensorCopy will call
    // mutable_data
    if (skip_update) {
      VLOG(4) << "Adam skip update";
      framework::TensorCopy(
          param,
          ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          &param_out);
      framework::TensorCopy(
          mom1,
          ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          &mom1_out);
      framework::TensorCopy(
          mom2,
          ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          &mom2_out);
      framework::TensorCopy(
          beta1_pow,
          beta1_pow.place(),
          ctx.template device_context<platform::DeviceContext>(),
          beta1_pow_out);
      framework::TensorCopy(
          beta2_pow,
          beta2_pow.place(),
          ctx.template device_context<platform::DeviceContext>(),
          beta2_pow_out);
      return;
    }

    PADDLE_ENFORCE_EQ(beta1_pow_out->numel(),
                      1,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong size, Expected beta1 pow "
                          "output size is 1, but received "
                          "value is:%d.",
                          beta1_pow_out->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out->numel(),
                      1,
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
    float epsilon = static_cast<float>(ctx.Attr<float>("epsilon"));
    if (ctx.HasInput("EpsilonTensor")) {
      auto* epsilon_tensor = ctx.Input<framework::Tensor>("EpsilonTensor");
      epsilon = static_cast<float>(GetAttrFromTensor(epsilon_tensor));
    }

    if (grad_var->IsType<framework::LoDTensor>()) {
      auto& grad = GET_DATA_SAFELY(
          ctx.Input<LoDTensor>("Grad"), "Input", "Grad", "Adam");
      getDataPointer<float>(grad, &grad_c, ctx);

      int r = xpu::adam(dev_ctx.x_context(),
                        grad_c,
                        mom1_ptr,
                        mom2_ptr,
                        param_ptr,
                        beta1_pow_ptr,
                        beta2_pow_ptr,
                        lr_ptr,
                        mom1_out_ptr,
                        mom2_out_ptr,
                        param_out_ptr,
                        beta1,
                        beta2,
                        epsilon,
                        param.numel());

      xpu_wait(dev_ctx.x_context()->xpu_stream);
      PADDLE_ENFORCE_EQ(
          r == xpu::Error_t::SUCCESS,
          true,
          platform::errors::External("XPU API return wrong value[%d],", r));

      freeData<float>(grad, grad_c);

      copyOutData<float>(xpu_mom1_out, &mom1_out, ctx);
      copyOutData<float>(xpu_mom2_out, &mom2_out, ctx);
      copyOutData<float>(xpu_param_out, &param_out, ctx);

      if (!use_global_beta_pow) {
        // update in cpu and then copy to xpu
        if (beta1_pow.place() == platform::CPUPlace() &&
            beta2_pow.place() == platform::CPUPlace()) {
          setBetaData(beta1_pow, beta1_pow_out, beta1);

          setBetaData(beta2_pow, beta2_pow_out, beta2);
        } else {
          float* beta1_pow_out_p1 = nullptr;

          if (beta1_pow_out->dtype() ==
              paddle::experimental::DataType::FLOAT16) {
            scale<DeviceContext, float>(
                beta1_pow_out, beta1_pow, beta1_pow_ptr, beta1, ctx);
          } else {
            beta1_pow_out_p1 =
                beta1_pow_out->mutable_data<float>(ctx.GetPlace());
            r = xpu::scale(dev_ctx.x_context(),
                           beta1_pow_ptr,
                           beta1_pow_out_p1,
                           beta1_pow.numel(),
                           false,
                           beta1,
                           0.0f);
            xpu_wait(dev_ctx.x_context()->xpu_stream);
            PADDLE_ENFORCE_EQ(
                r,
                xpu::SUCCESS,
                platform::errors::External(
                    "XPU kernel scale occur error in adam error code ",
                    r,
                    XPUAPIErrorMsg[r]));
          }

          float* beta2_pow_out_p1 = nullptr;
          if (beta2_pow_out->dtype() ==
              paddle::experimental::DataType::FLOAT16) {
            scale<DeviceContext, float>(
                beta2_pow_out, beta2_pow, beta2_pow_ptr, beta2, ctx);
          } else {
            const float* beta2_pow_data = beta2_pow.template data<float>();
            beta2_pow_out_p1 =
                beta2_pow_out->mutable_data<float>(ctx.GetPlace());
            r = xpu::scale(dev_ctx.x_context(),
                           beta2_pow_data,
                           beta2_pow_out_p1,
                           beta2_pow.numel(),
                           false,
                           beta2,
                           0.0f);
            xpu_wait(dev_ctx.x_context()->xpu_stream);
            PADDLE_ENFORCE_EQ(
                r,
                xpu::SUCCESS,
                platform::errors::External(
                    "XPU kernel scale occur error in adam error code ",
                    r,
                    XPUAPIErrorMsg[r]));
          }
        }
      }
    } else if (grad_var->IsType<phi::SelectedRows>()) {
      auto* grad = ctx.Input<phi::SelectedRows>("Grad");

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

      phi::SelectedRows tmp_grad_merge;
      const phi::SelectedRows* grad_merge_ptr;
      if (is_strict_sorted) {
        grad_merge_ptr = grad;
      } else {
        scatter::MergeAdd<platform::XPUDeviceContext, float> merge_func;
        merge_func(ctx.template device_context<platform::XPUDeviceContext>(),
                   *grad,
                   &tmp_grad_merge,
                   true);

        xpu_wait(dev_ctx.x_context()->xpu_stream);
        grad_merge_ptr = &tmp_grad_merge;
      }

      auto& grad_merge = *grad_merge_ptr;
      auto& grad_tensor = grad_merge.value();

      getDataPointer<float>(grad_tensor, &grad_c, ctx);

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
      memory::Copy(ctx.GetPlace(),
                   xpu_rows,
                   platform::CPUPlace(),
                   rows.data(),
                   row_count * sizeof(int));
      auto row_numel = grad_tensor.numel() / grad_merge.rows().size();
      auto ori_rows = param.numel() / row_numel;

      int lazy_mode = static_cast<int>(ctx.Attr<bool>("lazy_mode"));
      int r = xpu::sparse_adam(dev_ctx.x_context(),
                               grad_c,
                               mom1_ptr,
                               mom2_ptr,
                               param_ptr,
                               beta1_pow_ptr,
                               beta2_pow_ptr,
                               lr_ptr,
                               mom1_out_ptr,
                               mom2_out_ptr,
                               param_out_ptr,
                               beta1,
                               beta2,
                               epsilon,
                               ori_rows,
                               xpu_rows,
                               row_numel,
                               grad_merge.rows().size(),
                               lazy_mode);

      PADDLE_ENFORCE_EQ(
          r == xpu::Error_t::SUCCESS,
          true,
          platform::errors::External("XPU API return wrong value[%d],", r));

      freeData<float>(grad_tensor, grad_c);

      copyOutData<float>(xpu_mom1_out, &mom1_out, ctx);
      copyOutData<float>(xpu_mom2_out, &mom2_out, ctx);
      copyOutData<float>(xpu_param_out, &param_out, ctx);

      if (!use_global_beta_pow) {
        // update in cpu and then copy to xpu
        if (beta1_pow.place() == platform::CPUPlace() &&
            beta2_pow.place() == platform::CPUPlace()) {
          setBetaData(beta1_pow, beta1_pow_out, beta1);

          setBetaData(beta2_pow, beta2_pow_out, beta2);
        } else {
          float* beta1_pow_out_p1 = nullptr;

          if (beta1_pow_out->dtype() ==
              paddle::experimental::DataType::FLOAT16) {
            scale<DeviceContext, float>(
                beta1_pow_out, beta1_pow, beta1_pow_ptr, beta1, ctx);
          } else {
            const float* beta1_pow_data = beta1_pow.template data<float>();
            beta1_pow_out_p1 =
                beta1_pow_out->mutable_data<float>(ctx.GetPlace());
            r = xpu::scale(dev_ctx.x_context(),
                           beta1_pow_data,
                           beta1_pow_out_p1,
                           beta1_pow.numel(),
                           false,
                           beta1,
                           0.0f);
            xpu_wait(dev_ctx.x_context()->xpu_stream);
            PADDLE_ENFORCE_EQ(
                r,
                xpu::SUCCESS,
                platform::errors::External(
                    "XPU kernel scale occur error in adam error code ",
                    r,
                    XPUAPIErrorMsg[r]));
          }

          float* beta2_pow_out_p1 = nullptr;
          if (beta2_pow_out->dtype() ==
              paddle::experimental::DataType::FLOAT16) {
            scale<DeviceContext, float>(
                beta2_pow_out, beta2_pow, beta2_pow_ptr, beta2, ctx);
          } else {
            const float* beta2_pow_data = beta2_pow.template data<float>();
            beta2_pow_out_p1 =
                beta2_pow_out->mutable_data<float>(ctx.GetPlace());
            r = xpu::scale(dev_ctx.x_context(),
                           beta2_pow_data,
                           beta2_pow_out_p1,
                           beta2_pow.numel(),
                           false,
                           beta2,
                           0.0f);
            xpu_wait(dev_ctx.x_context()->xpu_stream);
            PADDLE_ENFORCE_EQ(
                r,
                xpu::SUCCESS,
                platform::errors::External(
                    "XPU kernel scale occur error in adam error code ",
                    r,
                    XPUAPIErrorMsg[r]));
          }
        }
      }
    } else {
      PADDLE_ENFORCE_EQ(1,
                        2,
                        platform::errors::InvalidArgument(
                            "Variable type not supported by adam_op"));
    }

    freeData<float>(param, param_ptr);
    freeData<float>(mom1, mom1_ptr);
    freeData<float>(mom2, mom2_ptr);
    freeData<float>(lr, lr_ptr);
  }
};
#endif

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
#ifdef PADDLE_WITH_XPU
REGISTER_OP_XPU_KERNEL(
    adam,
    ops::AdamOpXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::AdamOpXPUKernel<paddle::platform::XPUDeviceContext,
                         paddle::platform::float16>);
#endif
