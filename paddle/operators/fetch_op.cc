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
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    typedef std::vector<framework::Tensor> FetchOutputs;
    PADDLE_ENFORCE(ctx->HasInput("Input"), "Input should be not null.");
    int col = ctx->Attrs().Get<int>("col");
    framework::Variable* g_fetch_variable =
        framework::GetScope()->FindVar("fetch_value");

    FetchOutputs* tensors = g_fetch_variable->GetMutable<FetchOutputs>();
    if (tensors->size() < static_cast<size_t>(col + 1)) {
      tensors->resize(col + 1);
    }

    auto input_dim = ctx->GetInputDim("Input");
    framework::Tensor tmp;
    tmp.Resize(input_dim);
    (*tensors)[col].Resize(input_dim);

    // TODO(qijun) need to handle LodTensor later
  }

  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return static_cast<framework::DataType>(Attr<int>("data_type"));
  }
};

class FetchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FetchOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<int>("data_type", "output data type")
        .SetDefault(framework::DataType::FP32);
    AddAttr<int>("col", "The col in global fetch variable").SetDefault(0);
    AddAttr<std::vector<int>>("dims", "The dimension of fetch tensor.");
    AddInput("Input", "The output of fetch op.");
    AddComment(R"DOC(Fetch data to global fetch variable)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fetch, ops::FetchOp, ops::FetchOpMaker);
REGISTER_OP_CPU_KERNEL(fetch, ops::FetchKernel<float>);
