#include "paddle/operators/nccl/nccl_ops.h"

namespace paddle {
namespace operators {

// AllreduceOp
class NCCLAllReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  // allreduce do nothing in infershape
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            " Input(X) of AllReduce op input should not be NULL");
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    PADDLE_ENFORCE(ins.size() == outs.size(), "Input(X) and Output(Out) must have same size");
    for(size_t i=0; i < ins.size(); ++i) {
      outs[i]->Resize(ins[i]->dims());
    }
    std::string reduction = ctx.Attr<std::string>("reduction");
    PADDLE_ENFORCE( (reduction == "ncclSum" || reduction == "ncclProd" ||
                     reduction == "ncclMin" || reduction == "ncclMax"), "invalid reduction!");
  }
};

template <typename T>
class NCCLAllreduceOp : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *ctx = static_cast<NCCLContext *>(context.device_context());
  }
};

// BcastSendOp
template <typename T>
class NCCLBcastSendOp final : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            " Input(X) of BcastSend op input should not be NULL");
  }
};

// BcastRecvOp
template <typename T>
class NCCLBcastRecvOp final : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            " Input(X) of BcastRecv op input should not be NULL");
  }
};


class NCCLAllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
  NCCLAllReduceOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of AllReduce op");
    AddOutput("Out", "The output of AllReduce op");
    AddAttr<std::string>("reduction: {'min', 'max', 'prod', 'sum'}.");
    AddComment(R"DOC(
            AllReduce the input tensors.
        )DOC");
  }
};

class NCCLBcastSendOpMaker : public framework::OpProtoAndCheckerMaker {
  NCCLAllReduceOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of BcastSend op");
    AddComment(R"DOC(
            BcastSend the tensors.
        )DOC");
  }
};

class NCCLBcastRecvOpMaker : public framework::OpProtoAndCheckerMaker {
  NCCLAllReduceOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "The output of BcastRecv op");
    AddComment(R"DOC(
            BcastRecv the tensors.
        )DOC");
  }
};

}
}
