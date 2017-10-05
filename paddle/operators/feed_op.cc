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

#include "paddle/operators/feed_op.h"

namespace paddle {
namespace operators {

class FeedOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    typedef std::vector<framework::Tensor> FeedInputs;
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output should be not null.");
    int col = ctx->Attrs().Get<int>("col");
    framework::Variable* g_feed_variable =
        framework::GetScope()->FindVar("feed_value");
    FeedInputs tensors = g_feed_variable->Get<FeedInputs>();
    auto in_dim = tensors[col].dims();
    ctx->SetOutputDim("Y", in_dim);
    // need to handle LodTensor later
  }
};

class FeedOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FeedOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<int>("col", "The col in Global Feed Variable");
    AddOutput("Out", "The output of dropout op.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(feed, ops::FeedOp, ops::FeedOpMaker);
REGISTER_OP_CPU_KERNEL(feed, ops::FeedKernel<float>);
