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
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output should be not null.");
    auto& shape = ctx->Attrs().Get<std::vector<int>>("dims");
    std::vector<int64_t> shape_int64(shape.size(), 0);
    std::transform(shape.begin(), shape.end(), shape_int64.begin(),
                   [](int a) { return static_cast<int64_t>(a); });
    ctx->SetOutputDim("Out", framework::make_ddim(shape_int64));
    // TODO(qijun): need to handle LodTensor later
  }

  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return static_cast<framework::DataType>(Attr<int>("data_type"));
  }
};

class FeedOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FeedOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<int>("data_type", "output data type")
        .SetDefault(framework::DataType::FP32);
    AddAttr<int>("col", "The col in global feed variable").SetDefault(0);
    AddAttr<std::vector<int>>("dims", "The dimension of feed tensor.");
    AddOutput("Out", "The output of feed op.");
    AddComment(R"DOC(Feed data from global feed variable)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(feed, ops::FeedOp, ops::FeedOpMaker);
REGISTER_OP_CPU_KERNEL(feed, ops::FeedKernel<float>);
