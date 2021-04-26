#include <iostream>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/platform/place.h"
#include "unordered_set"

namespace paddle {
namespace operators {

class SoftmaxBlockSparseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");

    // return UnaryOpUnchangedInferShapeCheckAxis(ctx);
  }

 protected:
  // framework::OpKernelType GetExpectedKernelType(
  //     const framework::ExecutionContext& ctx) const override {
  //   auto data_type =
  //       OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X",
  //       "LayOutRowPtr", "LayOutColIndex");
  //   return framework::OpKernelType(data_type, ctx.device_context());
  // }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class SoftmaxBlockSparseGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class SoftmaxBlockSparseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input tensor.");
    AddInput("LayOutRowPtr", "Layout csr format row pointer.");
    AddInput("LayOutColIndex", "Layout csr format column index.");
    AddOutput("Out", "Softmax of input tensor.");
    AddComment(R"DOC(
Block Sparse Softmax Operator.
)DOC");
  }
};

template <typename T>
class SoftmaxBlockSparseGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("softmax_blocksparse_grad");
    op->SetInput("Out", this->Output("Out"));
    op->SetInput("LayOutRowPtr", this->Input("LayOutRowPtr"));
    op->SetInput("LayOutColIndex", this->Input("LayOutColIndex"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class SoftmaxBlockSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // auto* X = context.Input<framework::Tensor>("X");
    // auto* Out = context.Output<framework::Tensor>("Out");
  }
};

template <typename T>
class SoftmaxBlockSparseGradCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // auto* Out = context.Input<framework::Tensor>("Out");
    // auto* dOut =
    //     context.Input<framework::Tensor>(framework::GradVarName("Out"));
    // auto* dX =
    // context.Output<framework::Tensor>(framework::GradVarName("X"));
  }
};
}
}

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    softmax_blocksparse, ops::SoftmaxBlockSparseOp,
    ops::SoftmaxBlockSparseOpMaker,
    ops::SoftmaxBlockSparseGradOpMaker<paddle::framework::OpDesc>,
    ops::SoftmaxBlockSparseGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(softmax_blocksparse_grad, ops::SoftmaxBlockSparseGradOp);

REGISTER_OP_CPU_KERNEL(softmax_blocksparse,
                       ops::SoftmaxBlockSparseCPUKernel<float>);
REGISTER_OP_CPU_KERNEL(softmax_blocksparse_grad,
                       ops::SoftmaxBlockSparseGradCPUKernel<float>);