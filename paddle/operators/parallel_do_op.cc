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

#include <vector>
#include "paddle/framework/executor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

constexpr char kInputs[] = "inputs";
constexpr char kParameters[] = "parameters";
constexpr char kPlaces[] = "places";
constexpr char kParallelBlock[] = "parallel_block";
constexpr char kOutputs[] = "outputs";
constexpr char kParallelScopes[] = "sub_block";
// #define GRAD_SUFFIX "@GRAD"
// constexpr char kInputGrads[] = "inputs" GRAD_SUFFIX;
// constexpr char kOutputGrads[] = "outputs" GRAD_SUFFIX;
// constexpr char kParamGrads[] = "parameters" GRAD_SUFFIX;

using ParallelScopeVar = std::vector<framework::Scope *>;
using OperatorBase = framework::OperatorBase;

class ParallelDoOp : public OperatorBase {
 public:
  ParallelDoOp(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    // create scope
    // copy parameters
  }
};

class ParallelDoGradOp : public OperatorBase {
 public:
  ParallelDoGradOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {}
};

class ParallelDoOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ParallelDoOpProtoMaker(framework::OpProto *proto,
                         framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(kInputs, "").AsDuplicable();
    AddInput(kParameters, "").AsDuplicable();
    AddInput(kPlaces, "");
    AddOutput(kOutputs, "").AsDuplicable();
    AddOutput(kParallelScopes, "");
    AddAttr<framework::BlockDescBind *>(kParallelBlock, "");
    AddComment(R"DOC(
ParallelDo Operator.
)DOC");
  }
};

class ParallelDoGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  virtual std::unique_ptr<framework::OpDescBind> Apply() const {
    PADDLE_THROW("Not Implemented");
    auto *grad = new framework::OpDescBind();
    grad->SetType("recurrent_grad");
    for (auto &input_param : this->InputNames()) {
      grad->SetInput(input_param, this->Input(input_param));
      grad->SetOutput(framework::GradVarName(input_param),
                      this->InputGrad(input_param));
    }

    for (auto &output_param : this->OutputNames()) {
      if (output_param == kParallelScopes) {
        grad->SetInput(output_param, this->Output(output_param));
        grad->SetInput(framework::GradVarName(output_param),
                       this->Output(output_param));
      } else {
        grad->SetInput(output_param, this->Output(output_param));
        grad->SetInput(framework::GradVarName(output_param),
                       this->OutputGrad(output_param));
      }
    }
    grad->SetAttrMap(this->Attrs());
    grad->SetBlockAttr(kParallelBlock, *grad_block_[0]);

    return std::unique_ptr<framework::OpDescBind>(grad);
  }
};

class ParallelDoGradOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_THROW("Not Implemented");
    // std::vector<std::string> input{kInputs};
    // std::vector<std::string> output{kOutputs};
    // for (auto &s : input) {
    //   PADDLE_ENFORCE(ctx->HasInputs(s));
    //   PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName(s)),
    //                  "Cannot find the gradient variable %s",
    //                  framework::GradVarName(s));
    // }
    // for (auto &s : output) {
    //   PADDLE_ENFORCE(ctx->HasInputs(s));
    // }
    // for (auto &s : input) {
    //   ctx->SetOutputsDim(framework::GradVarName(s), ctx->GetInputsDim(s));
    // }
    // if (ctx->HasInputs(kParameters)) {
    //   PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName(kParameters)));
    //   ctx->SetOutputsDim(framework::GradVarName(kParameters),
    //                      ctx->GetInputsDim(kParameters));
    // }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(parallel_do, paddle::operators::ParallelDoOp,
                  paddle::operators::ParallelDoOpProtoMaker,
                  paddle::operators::ParallelDoGradOpDescMaker);
REGISTER_OPERATOR(parallel_do_grad, paddle::operators::ParallelDoGradOp,
                  paddle::operators::ParallelDoGradOpShapeInference);
