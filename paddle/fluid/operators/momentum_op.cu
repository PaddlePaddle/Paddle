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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/momentum_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void MomentumKernel(const T* p, const T* g, const T* v,
                               const T* learning_rate, const T mu,
                               const int64_t num, bool use_nesterov, T* p_out,
                               T* v_out) {
  T lr = learning_rate[0];
  if (use_nesterov) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += blockDim.x * gridDim.x) {
      T g_val = g[i];
      T v_new = v[i] * mu + g_val;
      v_out[i] = v_new;
      p_out[i] = p[i] - (g_val - v_new * mu) * lr;
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += blockDim.x * gridDim.x) {
      T v_new = v[i] * mu + g[i];
      v_out[i] = v_new;
      p_out[i] = p[i] - lr * v_new;
    }
  }
}

template <typename T>
class MomentumOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto velocity = ctx.Input<framework::Tensor>("Velocity");
    auto grad = ctx.Input<framework::Tensor>("Grad");
    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    T* p_out = param_out->mutable_data<T>(ctx.GetPlace());
    T* v_out = velocity_out->mutable_data<T>(ctx.GetPlace());

    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    auto* p = param->data<T>();
    auto* v = velocity->data<T>();
    auto* g = grad->data<T>();
    auto* lr = learning_rate->data<T>();

    int block = 512;
    int grid = (param->numel() + block - 1) / block;
    MomentumKernel<T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
        p, g, v, lr, mu, param->numel(), use_nesterov, p_out, v_out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(momentum, ops::MomentumOpCUDAKernel<float>,
                        ops::MomentumOpCUDAKernel<double>);
