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
      p_out[i] = p[i] - (g_val + v_new * mu) * lr;
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
__global__ void SparseMomentumKernel(const T* p, const T* g, const T* v,
                                     const T* lr, const T mu,
                                     const int64_t* grad_rows,
                                     const size_t grad_row_numel,
                                     const size_t grad_row_size,
                                     const T use_nesterov, T* p_out, T* v_out) {
  for (int i = blockIdx.x; i < grad_row_size; i += gridDim.x) {
    for (int j = threadIdx.x; j < grad_row_numel; j += blockDim.x) {
      size_t p_i = grad_rows[i] * grad_row_numel + j;
      size_t g_i = i * grad_row_numel + j;
      v_out[g_i] = v[g_i] * mu + g[g_i];
      if (use_nesterov) {
        p_out[p_i] = p[p_i] - (g[g_i] + v_out[g_i] * mu) * lr[0];
      } else {
        p_out[p_i] = p[p_i] - v_out[g_i] * lr[0];
      }
    }
  }
}

template <typename T>
class MomentumOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto* velocity_var = ctx.InputVar("Velocity");
    auto* grad_var = ctx.InputVar("Grad");

    if (grad_var->IsType<framework::LoDTensor>()) {
      PADDLE_ENFORCE(velocity_var->IsType<framework::LoDTensor>(),
                     "Unmatched Type of Param and Grad");
      auto velocity = ctx.Input<framework::Tensor>("Velocity");
      auto grad = ctx.Input<framework::Tensor>("Grad");
      auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
      T* p_out = param_out->mutable_data<T>(ctx.GetPlace());
      T* v_out = velocity_out->mutable_data<T>(ctx.GetPlace());
      auto* p = param->data<T>();
      auto* v = velocity->data<T>();
      auto* g = grad->data<T>();
      auto* lr = learning_rate->data<T>();

      const int kThreadPerBlock = 256;
      int grid = (param->numel() + kThreadPerBlock - 1) / kThreadPerBlock;
      MomentumKernel<
          T><<<grid, kThreadPerBlock, 0, ctx.cuda_device_context().stream()>>>(
          p, g, v, lr, mu, param->numel(), use_nesterov, p_out, v_out);
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      // sparse update embedding with selectedrows
      PADDLE_ENFORCE(velocity_var->IsType<framework::SelectedRows>(),
                     "Unmatched Type of Param and Grad");
      auto velocity = ctx.Input<framework::SelectedRows>("Velocity");
      auto grad = ctx.Input<framework::SelectedRows>("Grad");
      auto velocity_out = ctx.Output<framework::SelectedRows>("VelocityOut");

      // sparse update maybe empty.
      if (grad->rows().size() == 0) {
        return;
      }
      PADDLE_ENFORCE(grad->height() == velocity->height(),
                     "Unmatched gradient and velocity.");
      auto* p_out = param_out->mutable_data<T>(ctx.GetPlace());
      auto* v_out =
          velocity_out->mutable_value()->mutable_data<T>(ctx.GetPlace());
      auto* lr = learning_rate->data<T>();
      auto* p = param->data<T>();
      auto* g = grad->value().data<T>();
      auto* v = velocity->value().data<T>();
      size_t grad_row_numel = grad->value().numel() / grad->rows().size();
      size_t grad_row_size = grad->rows().size();
      framework::Vector<int64_t> rows(grad->rows());

      const int kThreadPerBlock = 256;
      int grid = (param->numel() + kThreadPerBlock - 1) / kThreadPerBlock;
      SparseMomentumKernel<
          T><<<grid, kThreadPerBlock, 0, ctx.cuda_device_context().stream()>>>(
          p, g, v, lr, mu, rows.CUDAData(ctx.GetPlace()), grad_row_numel,
          grad->rows().size(), use_nesterov, p_out, v_out);
    } else {
      PADDLE_THROW("Unsupported Variable Type of Grad");
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(momentum, ops::MomentumOpCUDAKernel<float>,
                        ops::MomentumOpCUDAKernel<double>);
