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

#include "gather.cu.h"
#include "paddle/operators/gather_op.h"
#include "scatter.cu.h"

namespace paddle {
namespace operators {

template <typename T>
class ScatterOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *Ref = ctx.Input<Tensor>("Ref");
    auto *Index = ctx.Input<Tensor>("Index");
    auto *Updates = ctx.Input<Tensor>("Updates");
    auto *Out = ctx.Output<Tensor>("Out");

    Out->ShareDataWith(*Ref);

    GPUScatterAssign<T>(ctx.device_context(), *Updates, *Index, Out);
  }
};

template <typename T>
class ScatterGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *dRef = ctx.Output<Tensor>(framework::GradVarName("Ref"));
    auto *dUpdates = ctx.Output<Tensor>(framework::GradVarName("Updates"));
    auto *Index = ctx.Input<Tensor>("Index");
    auto *dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));

    // In place gradient: dRef = dO
    dRef->ShareDataWith(*dOut);
    dUpdates->mutable_data<T>(ctx.GetPlace());
    // Gradient by Gather: dUpdates = dO[Index]
    GPUGather<T>(ctx.device_context(), *dOut, *Index, dUpdates);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(scatter, ops::ScatterOpCUDAKernel<float>);
REGISTER_OP_GPU_KERNEL(scatter_grad, ops::ScatterGradOpCUDAKernel<float>);
