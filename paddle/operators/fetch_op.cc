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

#include "paddle/operators/fetch_op.h"

namespace paddle {
namespace operators {

class FetchOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"), "Input should be not null.");
  }

  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return static_cast<framework::DataType>(Attr<int>("dataType"));
  }
};

class FetchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FetchOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<int>("dataType", "output data type")
        .SetDefault(framework::DataType::FP32);
    AddAttr<int>("col", "The col in global fetch variable").SetDefault(0);
    AddInput("Input", "The output of fetch op.");
    AddComment(R"DOC(Fetch data to global fetch variable)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fetch, ops::FetchOp, ops::FetchOpMaker);
REGISTER_OP_CPU_KERNEL(fetch, ops::FetchKernel<float>);
