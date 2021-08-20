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

#include <algorithm>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/pscore/distributed_push_dense_op.h"

namespace paddle {
namespace operators {

constexpr int64_t kNoPadding = -1;

class DistributedPushDenseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    return;
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class DistributedPushDenseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<int>("TableId", "(int, the table id of this embedding")
        .SetDefault(-1);
    AddAttr<std::vector<std::string>>("InputNames", "(vector, slot names")
        .SetDefault(std::vector<std::string>());
    AddComment(R"DOC(
Distributed Push Dense Operator.

push dense gradients to PSLib's Parameter Server.

The input gradients is all dense gradient tensors in a table.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(distributed_push_dense, ops::DistributedPushDenseOp,
                  ops::DistributedPushDenseOpMaker);

REGISTER_OP_CPU_KERNEL(distributed_push_dense,
                       ops::DistributedPushDenseKernel<
                           paddle::platform::CPUDeviceContext, float>);
