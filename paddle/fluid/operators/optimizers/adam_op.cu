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
#include "paddle/fluid/platform/gpu_launch_param_config.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void AdamKernel(T beta1, T beta2, T epsilon, const T* beta1_pow_,
                           const T* beta2_pow_, const T* moment1,
                           T* moment1_out, const T* moment2, T* moment2_out,
                           const T* lr_, const T* grad, const T* param,
                           T* param_out, int ndim, T* beta1_pow_out,
                           T* beta2_pow_out) {
  T lr = *lr_;
  T beta1_pow = *beta1_pow_;
  T beta2_pow = *beta2_pow_;

  lr *=
      sqrt(static_cast<T>(1.0) - beta2_pow) / (static_cast<T>(1.0) - beta1_pow);

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id == 0) {
    beta1_pow_out[0] = beta1_pow * beta1;
    beta2_pow_out[0] = beta2_pow * beta2;
  }

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    T p = param[id];
    T g = grad[id];
    T mom1 = moment1[id];
    T mom2 = moment2[id];
    mom1 = beta1 * mom1 + (static_cast<T>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<T>(1.0) - beta2) * g * g;
    p -= lr * (mom1 / (sqrt(mom2) + epsilon));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = p;
  }
}

template <typename T>
class AdamOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));

    using paddle::framework::LoDTensor;
    using paddle::operators::detail::Ref;

    int64_t min_row_size_to_use_multithread =
        ctx.Attr<int64_t>("min_row_size_to_use_multithread");
    bool lazy_mode = ctx.Attr<bool>("lazy_mode");
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    auto& param = Ref(ctx.Input<LoDTensor>("Param"), "Must set Param");
    auto* grad_var = ctx.InputVar("Grad");
    auto& mom1 = Ref(ctx.Input<LoDTensor>("Moment1"), "Must set Moment1");
    auto& mom2 = Ref(ctx.Input<LoDTensor>("Moment2"), "Must set Moment2");
    auto& lr =
        Ref(ctx.Input<LoDTensor>("LearningRate"), "Must set LearningRate");

    auto& beta1_pow =
        Ref(ctx.Input<LoDTensor>("Beta1Pow"), "Must set Beta1Pow");
    auto& beta2_pow =
        Ref(ctx.Input<LoDTensor>("Beta2Pow"), "Must set Beta2Pow");

    auto& param_out =
        Ref(ctx.Output<LoDTensor>("ParamOut"), "Must set ParamOut");
    auto& mom1_out =
        Ref(ctx.Output<LoDTensor>("Moment1Out"), "Must set Moment1Out");
    auto& mom2_out =
        Ref(ctx.Output<LoDTensor>("Moment2Out"), "Must set Moment1Out");
    auto& beta1_pow_out =
        Ref(ctx.Output<LoDTensor>("Beta1PowOut"), "Must set Beta1PowOut");
    auto& beta2_pow_out =
        Ref(ctx.Output<LoDTensor>("Beta2PowOut"), "Must set Beta2PowOut");

    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    if (ctx.HasInput("Beta1Tensor")) {
      auto* beta1_tensor = ctx.Input<framework::Tensor>("Beta1Tensor");
      beta1 = static_cast<T>(GetAttrFromTensor(beta1_tensor));
    }
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    if (ctx.HasInput("Beta2Tensor")) {
      auto* beta2_tensor = ctx.Input<framework::Tensor>("Beta2Tensor");
      beta2 = static_cast<T>(GetAttrFromTensor(beta2_tensor));
    }
    VLOG(3) << "beta1_pow.numel() : " << beta1_pow.numel()
            << "beta2_pow.numel() : " << beta2_pow.numel();
    VLOG(3) << "param.numel(): " << param.numel();

    if (grad_var->IsType<framework::LoDTensor>()) {
      auto& grad = Ref(ctx.Input<LoDTensor>("Grad"), "Must set Grad");

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      // update param and moment
      auto config = GetGpuLaunchConfig1D(dev_ctx, param.numel());

      // Fuse adam and beta compute together
      AdamKernel<T><<<config.block_per_grid.x, config.thread_per_block.x, 0,
                      dev_ctx.stream()>>>(
          beta1, beta2, epsilon, beta1_pow.template data<T>(),
          beta2_pow.template data<T>(), mom1.template data<T>(),
          mom1_out.template mutable_data<T>(ctx.GetPlace()),
          mom2.template data<T>(),
          mom2_out.template mutable_data<T>(ctx.GetPlace()),
          lr.template data<T>(), grad.template data<T>(),
          param.template data<T>(),
          param_out.template mutable_data<T>(ctx.GetPlace()), param.numel(),
          beta1_pow_out.template mutable_data<T>(ctx.GetPlace()),
          beta2_pow_out.template mutable_data<T>(ctx.GetPlace()));

    } else if (grad_var->IsType<framework::SelectedRows>()) {
      auto& grad =
          Ref(ctx.Input<framework::SelectedRows>("Grad"), "Must set Grad");
      if (grad.rows().size() == 0) {
        VLOG(3) << "grad row size is 0!!";
        return;
      }

      std::vector<int64_t> cpu_rows(grad.rows().begin(), grad.rows().end());
      bool is_strict_sorted = true;
      for (size_t i = 1; i < cpu_rows.size(); ++i) {
        if (cpu_rows[i - 1] >= cpu_rows[i]) {
          is_strict_sorted = false;
          break;
        }
      }

      framework::SelectedRows tmp_grad_merge;
      const framework::SelectedRows* grad_merge_ptr;
      if (is_strict_sorted) {
        grad_merge_ptr = &grad;
      } else {
        // merge duplicated rows if any.
        // The rows of grad_merge have been sorted inside MergeAdd functor
        scatter::MergeAdd<platform::CUDADeviceContext, T> merge_func;
        merge_func(ctx.template device_context<platform::CUDADeviceContext>(),
                   grad, &tmp_grad_merge, true);
        grad_merge_ptr = &tmp_grad_merge;
      }

      BetaPowFunctor<T> beta_functor(
          beta1, beta2, beta1_pow.template data<T>(),
          beta2_pow.template data<T>(),
          beta1_pow_out.template mutable_data<T>(ctx.GetPlace()),
          beta2_pow_out.template mutable_data<T>(ctx.GetPlace()));
      auto& grad_merge = *grad_merge_ptr;
      auto& grad_tensor = grad_merge.value();
      const T* grad_data = grad_tensor.template data<T>();
      const int64_t* rows = grad_merge.rows().Data(ctx.GetPlace());
      auto row_numel = grad_tensor.numel() / grad_merge.rows().size();

      SparseAdamFunctor<T, GPUAdam> functor(
          beta1, beta2, epsilon, beta1_pow.template data<T>(),
          beta2_pow.template data<T>(), mom1.template data<T>(),
          mom1_out.template mutable_data<T>(ctx.GetPlace()),
          mom2.template data<T>(),
          mom2_out.template mutable_data<T>(ctx.GetPlace()),
          lr.template data<T>(), grad_data, param.template data<T>(),
          param_out.template mutable_data<T>(ctx.GetPlace()), rows, row_numel,
          grad_merge.rows().size(), lazy_mode);

      // FIXME(minqiyang): remove BinarySearch in GPU later
      platform::ForRange<platform::CUDADeviceContext> for_range(
          static_cast<const platform::CUDADeviceContext&>(ctx.device_context()),
          param.numel());
      for_range(functor);
      // update beta1 and beta2
      platform::ForRange<platform::CUDADeviceContext> for_range_beta(
          static_cast<const platform::CUDADeviceContext&>(ctx.device_context()),
          beta2_pow.numel());
      for_range_beta(beta_functor);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Variable type not supported by adam_op"));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(adam, ops::AdamOpCUDAKernel<float>,
                        ops::AdamOpCUDAKernel<double>);
