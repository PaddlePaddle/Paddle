//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/ipu_runtime_op.h"

namespace paddle {
namespace operators {

class IpuRuntimeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.device_context());
  }
};

class IpuRuntimeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("FeedList", "FeedList of Graph").AsDuplicable();
    AddOutput("FetchList", "FetchList of Graph").AsDuplicable();
    AddAttr<int>("dtype",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::proto::VarType::FP32);
    AddComment(R"DOC(
Run graph by PopART runtime.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(ipu_runtime, ops::IpuRuntimeOp, ops::IpuRuntimeOpMaker);

REGISTER_OP_IPU_KERNEL(ipu_runtime, ops::IpuRuntimeKernel<float>,
                       ops::IpuRuntimeKernel<double>,
                       ops::IpuRuntimeKernel<int>,
                       ops::IpuRuntimeKernel<int64_t>,
                       ops::IpuRuntimeKernel<bool>,
                       ops::IpuRuntimeKernel<int8_t>,
                       ops::IpuRuntimeKernel<paddle::platform::float16>);
