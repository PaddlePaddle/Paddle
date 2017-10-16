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

#include "paddle/operators/nccl/nccl_ops.h"

namespace paddle {
namespace operators {

// AllreduceOp
class NCCLAllReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   " Input(X) of AllReduce op input should not be NULL");
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

// AllreduceOp
class NCCLAllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  NCCLAllReduceOpMaker(framework::OpProto *proto,
                       framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of AllReduce op");
    AddOutput("Out", "The output of AllReduce op");
    AddAttr<std::string>("reduction",
                         "{'ncclmin', 'ncclmax', 'ncclprod', 'ncclsum'}.");
    AddAttr<std::vector<int>>("gpus", "gpu id lists");
    AddComment(R"DOC(
            AllReduce the input tensors.
        )DOC");
  }
};

// BcastSendOp
class NCCLBcastSendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  NCCLAllReduceOpMaker(framework::OpProto *proto,
                       framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of BcastSend op");
    AddComment(R"DOC(
            BcastSend the tensors.
        )DOC");
  }
};

// BcastRecvOp
class NCCLBcastRecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  NCCLAllReduceOpMaker(framework::OpProto *proto,
                       framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "The output of BcastRecv op");
    AddComment(R"DOC(
            BcastRecv the tensors.
        )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(ncclAllReduce, ops::NCCLAllReduceOp,
                             ops::NCCLAllReduceOpMaker);
