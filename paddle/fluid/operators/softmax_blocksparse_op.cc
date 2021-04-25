#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "unordered_set"
#include <iostream>

namespace paddle {
namespace operators {

class SoftmaxBlockSparseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    return UnaryOpUnchangedInferShapeCheckAxis(ctx);
  }

 protected:
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

}
}

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(softmax_blocksparse,
                  ops::SoftmaxBlockSparseOp,
                  ops::SoftmaxBlockSparseOpMaker,
                  ops::SoftmaxBlockSparseGradOpMaker<paddle::framework::OpDesc>,
                  ops::SoftmaxBlockSparseGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(softmax_blocksparse_grad, ops::SoftmaxBlockSparseGradOp);