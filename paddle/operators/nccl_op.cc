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

#include "paddle/operators/nccl_op.h"

namespace paddle {
namespace operators {

// NCCLinitOp
class NCCLInitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Communicator"),
                   " Input(X) of AllReduce op input should not be NULL");
  }
};

class NCCLInitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  NCCLInitOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<std::vector<int>>("gpus", "gpu id lists");
    AddOutput("Communicator",
              "Create Communicator for communicating between gpus");
    AddComment(R"DOC(
               create communicator.
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
                   " Input(X) of AllReduce op input should not be NULL");

    auto x_dims = ctx->GetInputsDim("X");

    std::string reduction = ctx->Attrs().Get<std::string>("reduction");
    PADDLE_ENFORCE((reduction == "ncclSum" || reduction == "ncclProd" ||
                    reduction == "ncclMin" || reduction == "ncclMax"),
                   "invalid reduction.");

    ctx->SetOutputsDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
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

    ctx->SetOutputsDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

// BcastSendOp
class NCCLBcastSendOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   " Input(X) of Bcast op input should not be NULL");
    PADDLE_ENFORCE(ctx->HasInput("Communicator"),
                   " Input(Communicator) of Bcast op input should not be NULL");
  }
};

// BcastRecvOp
class NCCLBcastRecvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Communicator"),
                   " Input(Communicator) of Bcast op input should not be NULL");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   " Output(Out) of Bcast op output should not be NULL");
  }
};

// AllreduceOp
class NCCLAllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  NCCLAllReduceOpMaker(framework::OpProto *proto,
                       framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of AllReduce op");
    AddInput("Communicator", "Communicator for communicating between gpus");
    AddOutput("Out", "The output of AllReduce op");
    AddAttr<std::string>("reduction",
                         "{'ncclmin', 'ncclmax', 'ncclprod', 'ncclsum'}.");
    AddAttr<std::vector<int>>("gpus", "gpu id lists");
    AddComment(R"DOC(
            AllReduce the input tensors.
        )DOC");
  }
};

// BcastSend should be in the root
// BcastSendOp
class NCCLBcastSendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  NCCLAllBcastSendOpMaker(framework::OpProto *proto,
                          framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of BcastSend op");
    AddInput("Communicator", "Communicator for communicating between gpus");
    AddAttr<int>("root", "root gpu of Bcast");
    AddComment(R"DOC(
            Bcast the tensors.
        )DOC");
  }
};

// BcastOp
class NCCLBcastRecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  NCCLAllBcastRecvOpMaker(framework::OpProto *proto,
                          framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Communicator", "Communicator for communicating between gpus");
    AddAttr<int>("root", "root gpu of BcastRecv");
    AddOutput("Out", "The output of Bcast");
    AddComment(R"DOC(
            Bcast the tensors.
        )DOC");
  }
};

// BcastRecvOp
class NCCLReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  NCCLReduceOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of Reduce op");
    AddInput("Communicator", "Communicator for communicating between gpus");
    AddOutput("Out", "The output of Reduce op");
    AddComment(R"DOC(
            Reduce the tensors.
        )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(ncclAllReduce, ops::NCCLAllReduceOp,
                             ops::NCCLAllReduceOpMaker);
REGISTER_OP_WITHOUT_GRADIENT(ncclInit, ops::NCCLInitOp, ops::NCCLInitOpMaker);
REGISTER_OP_WITHOUT_GRADIENT(ncclBcastSend, ops::NCCLBcastSendOp,
                             ops::NCCLBcastSendOpMaker);
REGISTER_OP_WITHOUT_GRADIENT(ncclBcastRecv, ops::NCCLBcastRecvOp,
                             ops::NCCLBcastRecvOpMaker);
REGISTER_OP_WITHOUT_GRADIENT(ncclReduce, ops::NCCLReduceOp,
                             ops::NCCLReduceOpMaker);
REGISTER_OP_CPU_KERNEL(ncclInit, ops::NCCLInitKernel<float>);
