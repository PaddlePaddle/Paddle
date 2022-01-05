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

#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/optimizers/merged_adam_op.h"

namespace paddle {
namespace operators {

template <typename T, typename MT>
__global__ void AdamKernelREG(MT beta1, MT beta2, MT epsilon, MT beta1_pow_,
                              MT beta2_pow_, const MT* moment1, MT* moment1_out,
                              const MT* moment2, MT* moment2_out, const MT* lr_,
                              const T* grad, const T* param, T* param_out,
                              const MT* master_param, MT* master_param_out,
                              int ndim) {
  MT lr = *lr_;
  MT beta1_pow = beta1_pow_;
  MT beta2_pow = beta2_pow_;

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);
    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;

    MT denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    p += (mom1 / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T, typename MT>
__global__ void AdamKernelMEM(MT beta1, MT beta2, MT epsilon,
                              const MT* beta1_pow_, const MT* beta2_pow_,
                              const MT* moment1, MT* moment1_out,
                              const MT* moment2, MT* moment2_out, const MT* lr_,
                              const T* grad, const T* param, T* param_out,
                              const MT* master_param, MT* master_param_out,
                              int ndim) {
  MT lr = *lr_;
  MT beta1_pow = *beta1_pow_;
  MT beta2_pow = *beta2_pow_;

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);
    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;

    MT denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    p += (mom1 / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T>
__global__ void UpdateBetaPow(T beta1, T beta2, const T* beta1_pow_,
                              const T* beta2_pow_, T* beta1_pow_out,
                              T* beta2_pow_out) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
}

template <typename T>
class MergedAdamOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using MPDType = typename details::MPTypeTrait<T>::Type;

    auto param = ctx.MultiInput<framework::Tensor>("Param");
    auto grad = ctx.MultiInput<framework::Tensor>("Grad");
    auto lr = ctx.MultiInput<framework::Tensor>("LearningRate");
    auto mom1 = ctx.MultiInput<framework::Tensor>("Moment1");
    auto mom2 = ctx.MultiInput<framework::Tensor>("Moment2");
    auto beta1_pow = ctx.MultiInput<framework::Tensor>("Beta1Pow");
    auto beta2_pow = ctx.MultiInput<framework::Tensor>("Beta2Pow");

    auto param_out = ctx.MultiOutput<framework::Tensor>("ParamOut");
    auto mom1_out = ctx.MultiOutput<framework::Tensor>("Moment1Out");
    auto mom2_out = ctx.MultiOutput<framework::Tensor>("Moment2Out");
    auto beta1_pow_out = ctx.MultiOutput<framework::Tensor>("Beta1PowOut");
    auto beta2_pow_out = ctx.MultiOutput<framework::Tensor>("Beta2PowOut");

    MPDType beta1 = static_cast<MPDType>(ctx.Attr<float>("beta1"));
    MPDType beta2 = static_cast<MPDType>(ctx.Attr<float>("beta2"));
    MPDType epsilon = static_cast<MPDType>(ctx.Attr<float>("epsilon"));
    bool use_global_beta_pow = ctx.Attr<bool>("use_global_beta_pow");
    VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

    const bool multi_precision = ctx.Attr<bool>("multi_precision");
    auto master_param = ctx.MultiInput<framework::Tensor>("MasterParam");
    auto master_param_out =
        ctx.MultiOutput<framework::Tensor>("MasterParamOut");

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    size_t param_num = param.size();
    for (size_t idx = 0; idx < param_num; idx++) {
      const MPDType* master_in_data =
          multi_precision ? master_param[idx]->data<MPDType>() : nullptr;
      MPDType* master_out_data =
          multi_precision
              ? master_param_out[idx]->mutable_data<MPDType>(ctx.GetPlace())
              : nullptr;

      // update param and moment
      int threads = 512;
      int blocks = (param[idx]->numel() + threads - 1) / threads;

      if (beta1_pow[idx]->place() == platform::CPUPlace() &&
          beta2_pow[idx]->place() == platform::CPUPlace()) {
        // Compute with betapow in REG
        AdamKernelREG<T, MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1, beta2, epsilon, *beta1_pow[idx]->data<MPDType>(),
            *beta2_pow[idx]->data<MPDType>(), mom1[idx]->data<MPDType>(),
            mom1_out[idx]->mutable_data<MPDType>(ctx.GetPlace()),
            mom2[idx]->data<MPDType>(),
            mom2_out[idx]->mutable_data<MPDType>(ctx.GetPlace()),
            lr[idx]->data<MPDType>(), grad[idx]->data<T>(),
            param[idx]->data<T>(),
            param_out[idx]->mutable_data<T>(ctx.GetPlace()), master_in_data,
            master_out_data, param[idx]->numel());
        if (!use_global_beta_pow) {
          // Cpu update
          beta1_pow_out[idx]->mutable_data<MPDType>(platform::CPUPlace())[0] =
              beta1 * beta1_pow[idx]->data<MPDType>()[0];
          beta2_pow_out[idx]->mutable_data<MPDType>(platform::CPUPlace())[0] =
              beta2 * beta2_pow[idx]->data<MPDType>()[0];
        }
      } else {
        AdamKernelMEM<T, MPDType><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1, beta2, epsilon, beta1_pow[idx]->data<MPDType>(),
            beta2_pow[idx]->data<MPDType>(), mom1[idx]->data<MPDType>(),
            mom1_out[idx]->mutable_data<MPDType>(ctx.GetPlace()),
            mom2[idx]->data<MPDType>(),
            mom2_out[idx]->mutable_data<MPDType>(ctx.GetPlace()),
            lr[idx]->data<MPDType>(), grad[idx]->data<T>(),
            param[idx]->data<T>(),
            param_out[idx]->mutable_data<T>(ctx.GetPlace()), master_in_data,
            master_out_data, param[idx]->numel());
        if (!use_global_beta_pow) {
          // Update with gpu
          UpdateBetaPow<MPDType><<<1, 32, 0, dev_ctx.stream()>>>(
              beta1, beta2, beta1_pow[idx]->data<MPDType>(),
              beta2_pow[idx]->data<MPDType>(),
              beta1_pow_out[idx]->mutable_data<MPDType>(ctx.GetPlace()),
              beta2_pow_out[idx]->mutable_data<MPDType>(ctx.GetPlace()));
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(merged_adam, ops::MergedAdamOpCUDAKernel<float>,
                        ops::MergedAdamOpCUDAKernel<double>,
                        ops::MergedAdamOpCUDAKernel<plat::float16>);
