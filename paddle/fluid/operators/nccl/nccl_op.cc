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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"

namespace paddle {
namespace operators {

static constexpr char kParallelScopes[] = "parallel_scopes";

// NCCLinitOp
class NCCLInitOp : public framework::OperatorBase {
 public:
  NCCLInitOp(const std::string &type, const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(Input(kParallelScopes)),
                            "Can not find variable '%s' in the scope.",
                            kParallelScopes);
    const auto &name = Output("Communicator");
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(name),
                            "Can not find variable '%s' in the scope.", name);
    // A parallel do may not use all the gpus. For example, the batch size is 7
    // in the last batch while we have 8 gpu. In this case, parallel_do will
    // create 7 parallel scopes, so should ncclInitOp create 7 gpu peers
    auto &parallel_scopes = scope.FindVar(Input(kParallelScopes))
                                ->Get<std::vector<framework::Scope *>>();
    std::vector<int> gpus(parallel_scopes.size());
    for (int i = 0; i < static_cast<int>(parallel_scopes.size()); ++i) {
      gpus[i] = i;
    }
    PADDLE_ENFORCE(!gpus.empty(), "NCCL init with 0 gpus.");

    if (scope.FindVar(name) == nullptr) {
      PADDLE_THROW("Output(Communicator) is needed for ncclInit operator.");
    }

    platform::Communicator *comm =
        scope.FindVar(name)->GetMutable<platform::Communicator>();
    comm->InitAll(gpus);
  }
};

class NCCLInitOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto out_var_name = ctx->Output("Communicator").front();
    ctx->SetType(out_var_name, framework::proto::VarType::RAW);
  }
};

class NCCLInitOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {}
};

class NCCLInitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kParallelScopes, "The working place of parallel do.");
    AddOutput("Communicator",
              "Create Communicator for communicating between gpus");
    AddComment(R"DOC(
NCCLInit Operator.

Create communicator.

)DOC");
  }
};

// AllReduceOp
class NCCLAllReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   " Input(X) of AllReduce op input should not be NULL");
    PADDLE_ENFORCE(
        ctx->HasInput("Communicator"),
        " Input(Communicator) of AllReduce op input should not be NULL");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   " Output(Out) of AllReduce op output should not be NULL");
    std::string reduction = ctx->Attrs().Get<std::string>("reduction");
    PADDLE_ENFORCE((reduction == "ncclSum" || reduction == "ncclProd" ||
                    reduction == "ncclMin" || reduction == "ncclMax"),
                   "invalid reduction.");

    auto x_dims = ctx->GetInputsDim("X");
    ctx->SetOutputsDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

// AllReduceOp
class NCCLAllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of AllReduce op");
    AddInput("Communicator", "Communicator for communicating between gpus");
    AddOutput("Out", "The output of AllReduce op");
    AddAttr<std::string>("reduction",
                         "(string, default 'ncclSum') "
                         "{'ncclMin', 'ncclMax', 'ncclProd', 'ncclSum'}.")
        .SetDefault("ncclSum");
    AddComment(R"DOC(
NCCLAllReduce Operator.

AllReduce the input tensors.

)DOC");
  }
};

// ReduceOp
class NCCLReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   " Input(X) of Reduce op input should not be NULL");
    PADDLE_ENFORCE(
        ctx->HasInput("Communicator"),
        " Input(Communicator) of Reduce op input should not be NULL");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   " Input(X) of Reduce op input should not be NULL");

    std::string reduction = ctx->Attrs().Get<std::string>("reduction");
    PADDLE_ENFORCE((reduction == "ncclSum" || reduction == "ncclProd" ||
                    reduction == "ncclMin" || reduction == "ncclMax"),
                   "invalid reduction.");

    auto x_dims = ctx->GetInputsDim("X");
    ctx->SetOutputsDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

// ReduceOp
class NCCLReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of Reduce op");
    AddInput("Communicator", "Communicator for communicating between gpus");
    AddOutput("Out", "The output of Reduce op");
    AddAttr<std::string>("reduction",
                         "(string, default 'ncclSum') "
                         "{'ncclMin', 'ncclMax', 'ncclProd', 'ncclSum'}.")
        .SetDefault("ncclSum");
    AddAttr<int>("root",
                 "(int, default kInvalidGPUId) "
                 "Root gpu of the parameter. If not, "
                 "set(platform::kInvalidGPUId). Hashed by name.")
        .SetDefault(platform::kInvalidGPUId);
    AddComment(R"DOC(
NCCLReduce Operator.

Reduce the tensors.

)DOC");
  }
};

// BcastOp
class NCCLBcastOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   " Input(X) of Bcast op input should not be NULL");
    PADDLE_ENFORCE(ctx->HasInput("Communicator"),
                   " Input(Communicator) of Bcast op input should not be NULL");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   " Output(Out) of Bcast op output should not be NULL");

    int root = ctx->Attrs().Get<int>("root");
    PADDLE_ENFORCE(root != platform::kInvalidGPUId, "Bcast root must be set.");

    auto x_dims = ctx->GetInputsDim("X");
    ctx->SetOutputsDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

// BcastOp
class NCCLBcastOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of BcastSend op");
    AddInput("Communicator", "Communicator for communicating between gpus");
    AddOutput("Out", "The output of Bcast");
    AddAttr<int>("root",
                 "(int, default kInvalidGPUId) "
                 "Root gpu of the parameter. If not, "
                 "set(platform::kInvalidGPUId). Hashed by name.")
        .SetDefault(platform::kInvalidGPUId);
    AddComment(R"DOC(
NCCLBcast Operator.

Bcast the tensors.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(ncclInit, ops::NCCLInitOp,
                  paddle::framework::EmptyGradOpMaker, ops::NCCLInitOpMaker,
                  ops::NCCLInitOpVarTypeInference,
                  ops::NCCLInitOpShapeInference);

REGISTER_OP_WITHOUT_GRADIENT(ncclAllReduce, ops::NCCLAllReduceOp,
                             ops::NCCLAllReduceOpMaker);
REGISTER_OP_WITHOUT_GRADIENT(ncclBcast, ops::NCCLBcastOp,
                             ops::NCCLBcastOpMaker);
REGISTER_OP_WITHOUT_GRADIENT(ncclReduce, ops::NCCLReduceOp,
                             ops::NCCLReduceOpMaker);
