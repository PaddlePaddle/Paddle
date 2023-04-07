/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

class NopOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  }
};

class NopOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) The input tensor of nop op.").AsDuplicable();
    AddOutput("Out", "(Tensor) The output tensor of nop op.").AsDuplicable();
    AddComment(R"DOC(
Nop Operator

Do nothing, except let the input and output tensors occupy the memory and
establish the dependency between input and output tensors.
)DOC");
  }
};

template <typename T>
class NopKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(nop, ops::NopOp, ops::NopOpMaker);

REGISTER_OP_CPU_KERNEL(nop, ops::NopKernel<float>);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
REGISTER_OP_CUDA_KERNEL(nop,
                        ops::NopKernel<float>,
                        ops::NopKernel<phi::dtype::float16>,
                        ops::NopKernel<phi::dtype::bfloat16>);
#endif

REGISTER_OP_NPU_KERNEL(nop, ops::NopKernel<float>);
