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

#define EIGEN_USE_GPU
#include "paddle/fluid/operators/sgd_group_op.h"
#include "paddle/fluid/platform/cuda_helper.h"

namespace paddle {
namespace operators {

namespace {

template <typename T>
__global__ void SGDGroupKernel(const T* g, const T* p, const T* learning_rate,
                               const int num, T* p_out) {
  T lr = learning_rate[0];
  int grid_size = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += grid_size) {
    T g_data = g[i];
    T p_data = p[i];
    p_out[i] = p_data - lr * g_data;
  }
}

}  // namespace

template <typename T>
class SGDGroupOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto params = ctx.MultiInput<framework::Tensor>("Params");
    auto learning_rates = ctx.MultiInput<framework::Tensor>("LearningRates");
    auto grads = ctx.MultiInput<framework::Tensor>("Grads");

    auto param_outs = ctx.MultiOutput<framework::Tensor>("ParamOuts");

    auto grad_var = ctx.MultiInputVar("Grads");

    if (grad_var[0]->IsType<framework::LoDTensor>()) {
      for (size_t j = 0; j < params.size(); ++j) {
        auto* param_out_data = param_outs[j]->mutable_data<T>(ctx.GetPlace());
        auto* grad_data = grads[j]->data<T>();
        auto* param_data = params[j]->data<T>();
        int param_num = params[j]->numel();
        int block = 512;
        int grid = (param_num + block - 1) / block;
        SGDGroupKernel<
            T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
            grad_data, param_data, learning_rates[j]->data<T>(), param_num,
            param_out_data);
      }
    } else {
      PADDLE_THROW("Unsupported Variable Type of Grad");
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(sgd_group, ops::SGDGroupOpCUDAKernel<float>,
                        ops::SGDGroupOpCUDAKernel<double>);
