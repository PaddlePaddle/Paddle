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
#include "chunk_eval_op.h"
#include "paddle/framework/executor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"
#include "paddle/platform/place.h"

namespace paddle {
namespace operators {

constexpr char kInputs[] = "inputs";
constexpr char kParameters[] = "parameters";
constexpr char kPlaces[] = "places";

constexpr char kOutputs[] = "outputs";
constexpr char kParallelScopes[] = "parallel_scopes";

constexpr char kParallelBlock[] = "sub_block";

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
    auto *block = Attr<framework::BlockDescBind *>(kParallelBlock);
    auto *program = block->Program();

    // TODO(tonyyang-svail): get places from input
    std::vector<platform::Place> places;
    places.emplace_back(platform::CPUPlace());
    places.emplace_back(platform::CPUPlace());

    std::vector<framework::Scope *> sub_scopes;
    for (int place_idx = 0; place_idx < places.size(); ++place_idx) {
      VLOG(3) << "Run " << place_idx;

      sub_scopes.push_back(&scope.NewScope());

      auto &place = places[place_idx];
      auto *cur_scope = sub_scopes[place_idx];

      // copy parameter
      if (dev_ctx.GetPlace() != place) {
        PADDLE_THROW("Not Implemented");
      }

      // feed input
      for (auto &argu : Inputs(kInputs)) {
        auto *var = scope.FindVar(argu);
        const auto &tensor = var->Get<LoDTensor>();
        if (!tensor.lod().empty()) {
          PADDLE_THROW("Disable parallel lod for now");
        } else {
          PADDLE_ENFORCE(tensor.dims()[0] % places.size() == 0,
                         "Batch size should be divided by places size");
          int begin = place_idx * tensor.dims()[0] / places.size();
          int end = (place_idx + 1) * tensor.dims()[0] / places.size();
          auto feed_tensor = tensor.Slice(begin, end);
          feed_tensor.switch_place(place);

          auto *cur_var = cur_scope->Var(argu);
          auto *cur_tensor = cur_var->GetMutable<Tensor>();
          *cur_tensor = feed_tensor;
        }
      }

      // execute
      auto executor = framework::Executor(place);
      executor.Run(*program, cur_scope, block->ID(),
                   false /*create_local_scope*/);
    }

    // merge output
    for (auto &o_name : Outputs(kOutputs)) {
      std::vector<const framework::LoDTensor *> lod_tensors;
      for (auto *sub_scope : sub_scopes) {
        lod_tensors.push_back(&sub_scope->FindVar(o_name)->Get<LoDTensor>());
      }

      auto *lod_tensor_to_be_merged =
          scope.FindVar(o_name)->GetMutable<LoDTensor>();
      lod_tensor_to_be_merged->MergeLoDTensor(lod_tensors, dev_ctx.GetPlace());
    }
  }
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

class ParallelDoGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  virtual std::unique_ptr<framework::OpDescBind> Apply() const {
    auto *grad = new framework::OpDescBind();
    grad->SetType("parallel_do_grad");
    for (auto &input_param : this->InputNames()) {
      LOG(INFO) << input_param;
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
    std::vector<std::string> input{kParameters, kInputs};
    std::vector<std::string> output{kOutputs};
    for (auto &s : input) {
      PADDLE_ENFORCE(ctx->HasInputs(s));
      PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName(s)),
                     "Cannot find the gradient variable %s",
                     framework::GradVarName(s));
    }
    for (auto &s : output) {
      PADDLE_ENFORCE(ctx->HasInputs(s));
    }
    for (auto &s : input) {
      ctx->SetOutputsDim(framework::GradVarName(s), ctx->GetInputsDim(s));
    }
    if (ctx->HasInputs(kParameters)) {
      PADDLE_ENFORCE(ctx->HasOutputs(framework::GradVarName(kParameters)));
      ctx->SetOutputsDim(framework::GradVarName(kParameters),
                         ctx->GetInputsDim(kParameters));
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(parallel_do, paddle::operators::ParallelDoOp,
                  paddle::operators::ParallelDoOpProtoMaker,
                  paddle::operators::ParallelDoGradOpDescMaker);
REGISTER_OPERATOR(parallel_do_grad, paddle::operators::ParallelDoGradOp,
                  paddle::operators::ParallelDoGradOpShapeInference);
