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
#include "paddle/framework/threadpool.h"

namespace paddle {
namespace operators {

static constexpr char kInputs[] = "inputs";
static constexpr char kParameters[] = "parameters";
static constexpr char kPlaces[] = "places";

static constexpr char kOutputs[] = "outputs";
static constexpr char kParallelScopes[] = "parallel_scopes";

static constexpr char kParallelBlock[] = "sub_block";

// using ParallelScopeVar = std::vector<framework::Scope *>;
using LoDTensor = framework::LoDTensor;
using OperatorBase = framework::OperatorBase;

void SplitTensorAndMoveTensorToScopes(
    const framework::Scope &scope,
    const std::vector<framework::Scope *> &sub_scopes,
    const std::vector<platform::Place> &places,
    const std::vector<std::string> &names) {
  PADDLE_ENFORCE_EQ(sub_scopes.size(), places.size());
  for (auto &argu : names) {
    auto *var = scope.FindVar(argu);
    const auto &tensor = var->Get<LoDTensor>();
    auto lod_tensors = tensor.SplitLoDTensor(places);

    for (auto &lod : lod_tensors) {
      VLOG(3) << lod.dims();
    }

    for (size_t i = 0; i < sub_scopes.size(); ++i) {
      *sub_scopes[i]->Var(argu)->GetMutable<LoDTensor>() = lod_tensors[i];
    }
  }
}

void WaitOnPlaces(const std::vector<platform::Place> places) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();

  for (auto &place : places) {
    auto &dev_ctx = *pool.Get(place);
    dev_ctx.Wait();
  }
}

class ParallelDoOp : public framework::OperatorBase {
 public:
  ParallelDoOp(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::Place &place) const override {
    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    auto *block = Attr<framework::BlockDesc *>(kParallelBlock);
    auto *program = block->Program();

    auto &places = scope.FindVar(Input(kPlaces))->Get<platform::PlaceList>();

    auto &sub_scopes = *scope.FindVar(Output(kParallelScopes))
                            ->GetMutable<std::vector<framework::Scope *>>();
    for (size_t place_idx = 0; place_idx < places.size(); ++place_idx) {
      sub_scopes.push_back(&scope.NewScope());
    }

    // split input
    SplitTensorAndMoveTensorToScopes(scope, sub_scopes, places,
                                     Inputs(kInputs));
    // copy parameter
    for (auto &param : Inputs(kParameters)) {
      PADDLE_ENFORCE(scope.FindVar(param)->IsType<LoDTensor>(),
                     "Only support parameter type as LoDTensor");
      auto &src = scope.FindVar(param)->Get<LoDTensor>();
      for (size_t i = 0; i < places.size(); ++i) {
        auto &place = places[i];
        auto *sub_scope = sub_scopes[i];
        auto *dst = sub_scope->Var(param)->GetMutable<LoDTensor>();
        framework::Copy(src, place, dst);
      }
    }
    WaitOnPlaces(places);

    std::vector<std::future<void>> workers;
    workers.reserve(places.size());
    for (size_t place_idx = 0; place_idx < places.size(); ++place_idx) {
      VLOG(3) << "Run " << place_idx;

      auto &place = places[place_idx];
      auto *cur_scope = sub_scopes[place_idx];

      workers.emplace_back(framework::Async([program, cur_scope, place, block] {
        framework::Executor executor(place);
        executor.Run(*program, cur_scope, block->ID(),
                     false /*create_local_scope*/);
      }));
    }
    for (auto &worker : workers) {
      worker.wait();
    }
    WaitOnPlaces(places);

    // merge output
    for (auto &o_name : Outputs(kOutputs)) {
      std::vector<const framework::LoDTensor *> lod_tensors;
      lod_tensors.reserve(sub_scopes.size());
      for (auto *sub_scope : sub_scopes) {
        lod_tensors.emplace_back(&sub_scope->FindVar(o_name)->Get<LoDTensor>());
      }

      auto *lod_tensor_to_be_merged =
          scope.FindVar(o_name)->GetMutable<LoDTensor>();
      lod_tensor_to_be_merged->MergeLoDTensor(lod_tensors, dev_ctx.GetPlace());
    }
    WaitOnPlaces(places);
  }
};

class ParallelDoOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ParallelDoOpProtoMaker(OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(kInputs, "").AsDuplicable();
    AddInput(kParameters, "").AsDuplicable();
    AddInput(kPlaces, "");
    AddOutput(kOutputs, "").AsDuplicable();
    AddOutput(kParallelScopes, "");
    AddAttr<framework::BlockDesc *>(kParallelBlock, "");
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
           const platform::Place &place) const override {
    // // get device context from pool
    // platform::DeviceContextPool &pool =
    //        platform::DeviceContextPool::Instance();
    // auto &dev_ctx = *pool.Get(place);

    auto *block = Attr<framework::BlockDesc *>(kParallelBlock);
    auto *program = block->Program();

    auto &sub_scopes = scope.FindVar(Input(kParallelScopes))
                           ->Get<std::vector<framework::Scope *>>();

    auto &places = scope.FindVar(Input(kPlaces))->Get<platform::PlaceList>();

    // feed output@grad
    SplitTensorAndMoveTensorToScopes(scope, sub_scopes, places,
                                     Inputs(framework::GradVarName(kOutputs)));
    WaitOnPlaces(places);

    // for debugging
    for (auto &s : Inputs(framework::GradVarName(kOutputs))) {
      VLOG(3) << s;
      VLOG(3) << scope.FindVar(s)->Get<LoDTensor>();
      for (auto *sub_scope : sub_scopes) {
        VLOG(3) << sub_scope->FindVar(s)->Get<LoDTensor>();
      }
    }

    // exe run
    std::vector<std::future<void>> workers;
    for (size_t place_idx = 0; place_idx < places.size(); ++place_idx) {
      VLOG(3) << "Run " << place_idx;

      auto &place = places[place_idx];
      auto *cur_scope = sub_scopes[place_idx];

      // execute
      workers.emplace_back(framework::Async([program, cur_scope, place, block] {
        framework::Executor executor(place);
        executor.Run(*program, cur_scope, block->ID(),
                     false /*create_local_scope*/);
      }));
    }
    for (auto &worker : workers) {
      worker.wait();
    }
    WaitOnPlaces(places);

    // merge grad
    for (auto &s : Outputs(framework::GradVarName(kParameters))) {
      VLOG(3) << "merge grad " << s;

      auto &t = sub_scopes[0]->FindVar(s)->Get<LoDTensor>();
      VLOG(3) << t;

      std::string s_buf = s + "@BUF";
      auto *t_buf = sub_scopes[0]->Var(s_buf)->GetMutable<LoDTensor>();

      for (size_t place_idx = 1; place_idx < places.size(); ++place_idx) {
        auto &tt = sub_scopes[place_idx]->FindVar(s)->Get<LoDTensor>();
        VLOG(3) << place_idx;
        VLOG(3) << tt;
        framework::Copy(tt, places[0], t_buf);

        auto sum_op = framework::OpRegistry::CreateOp(
            "sum", {{"X", {s, s_buf}}}, {{"Out", {s}}},
            framework::AttributeMap{});
        sum_op->Run(*sub_scopes[0], places[0]);
        WaitOnPlaces(places);
      }

      VLOG(3) << t;
      framework::Copy(t, place, scope.FindVar(s)->GetMutable<LoDTensor>());
    }
  }
};

class ParallelDoGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  virtual std::unique_ptr<framework::OpDesc> Apply() const {
    auto *grad = new framework::OpDesc();
    grad->SetType("parallel_do_grad");
    for (auto &input_param : this->InputNames()) {
      VLOG(3) << input_param;
      grad->SetInput(input_param, this->Input(input_param));
      if (input_param != kPlaces) {
        grad->SetOutput(framework::GradVarName(input_param),
                        this->InputGrad(input_param, false));
      }
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

    return std::unique_ptr<framework::OpDesc>(grad);
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
