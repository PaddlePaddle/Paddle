// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/share_buffer_op.h"

namespace paddle {
namespace operators {

class ShareBufferOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // dtype is not important
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    return expected_kernel_type;
  }
};

class ShareBufferOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensors of share buffer op")
        .AsDuplicable();
    AddOutput("Out", "(Tensor), The output tensors of share buffer op")
        .AsDuplicable();
    AddOutput("XOut",
              "(Tensor), The output tensors which are the same as X. It is "
              "used to build the graph dependency")
        .AsDuplicable();
    AddAttr<std::vector<bool>>("share_dims_and_dtype",
                               "Whether to share dims and data type")
        .SetDefault(std::vector<bool>());
    AddComment(
        R"DOC(Operator used to perform inplace memory reuse. It should be not exposed to Python APIs.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(share_buffer, ops::ShareBufferOp, ops::ShareBufferOpMaker);

// dtype is not important
REGISTER_OP_CPU_KERNEL(share_buffer, ops::ShareBufferOpKernel<float>);
