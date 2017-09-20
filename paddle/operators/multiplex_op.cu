/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
class MultiplexGPUKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto ins = ctx.MultiInput<Tensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto rows = ins[1]->dims()[0];
    auto cols = ins[1]->dims()[1];
    // copy index to cpu
    Tensor index_t_cpu;
    index_t_cpu.CopyFrom<T>(*(ins[0]), paddle::platform::CPUPlace());
    auto index = index_t_cpu.data<T>();
    for (auto i = 0; i < rows; i++) {
      int k = (int)index[i] + 1;
      cudaMemcpy(out->data<T>() + i * cols, ins[k]->data<T>() + i * cols,
                 cols * sizeof(T), cudaMemcpyDeviceToDevice);
    }
  }
};

template <typename T>
class MultiplexGradGPUKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<Tensor>("X");
    auto d_ins = ctx.MultiOutput<Tensor>(framework::GradVarName("X"));
    for (size_t i = 1; i < d_ins.size(); ++i) {
      if (d_ins[i]) {
        d_ins[i]->mutable_data<T>(ctx.GetPlace());
        auto dims = d_ins[i]->dims();
        cudaMemset(d_ins[i]->data<T>(), 0,
                   framework::product(dims) * sizeof(T));
      }
    }

    auto rows = ins[1]->dims()[0];
    auto cols = ins[1]->dims()[1];
    // copy index to cpu
    Tensor index_t_cpu;
    index_t_cpu.CopyFrom<T>(*(ins[0]), paddle::platform::CPUPlace());
    auto index = index_t_cpu.data<T>();
    for (auto i = 0; i < rows; i++) {
      int k = (int)index[i] + 1;
      if (d_ins[k]) {
        cudaMemcpy(d_ins[k]->data<T>() + i * cols, d_out->data<T>() + i * cols,
                   cols * sizeof(T), cudaMemcpyDeviceToDevice);
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_GPU_KERNEL(multiplex, ops::MultiplexGPUKernel<float>);
REGISTER_OP_GPU_KERNEL(multiplex_grad, ops::MultiplexGradGPUKernel<float>);
