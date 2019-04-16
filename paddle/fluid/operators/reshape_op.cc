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

#include "paddle/fluid/operators/reshape_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#ifdef PADDLE_WITH_MKLDNN
#include "mkldnn.hpp"
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

class ReshapeOp : public framework::OperatorWithKernel {
 public:
  ReshapeOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ReshapeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ReshapeOp should not be null.");

    const std::vector<int> &shape = ctx->Attrs().Get<std::vector<int>>("shape");
    PADDLE_ENFORCE(!shape.empty(),
                   "The shape information must be set by Attr(shape).");

    if (ctx->HasInput("Shape") && ctx->IsRuntime()) {
      // If true, set the shape of Output(Out) according to Input(Shape) in
      // ReshapeKernel with ExecutionContext. Also check LoD in ReshapeKernel.
      ctx->ShareLoD("X", /*->*/ "Out");
      return;
    }

    auto x_dims = ctx->GetInputDim("X");
    auto out_dims = ValidateShape(shape, x_dims);
    ctx->SetOutputDim("Out", out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = ctx.Input<framework::Tensor>("X")->type();
#ifdef PADDLE_WITH_MKLDNN
    return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                   framework::DataLayout::kMKLDNN,
                                   framework::LibraryType::kMKLDNN);
#endif
    return framework::OpKernelType(input_data_type, ctx.device_context());
  }
};

class ReshapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor). The input tensor of reshape operator.");
    AddInput("Shape",
             "(Tensor<int32>, optional). If provided, reshape according to "
             "this given shape. That is to say it has a higher priority than "
             "the shape attribute, while the shape attribute still should be "
             "set correctly to gurantee shape inference in compile time.")
        .AsDispensable();
    AddOutput("Out", "(Tensor). The output tensor of reshape operator.");
    AddAttr<std::vector<int>>(
        "shape", "(std::vector<int>) Target shape of reshape operator.");
    AddComment(R"DOC(
Reshape Operator.

Reshape Input(X) into the shape specified by Attr(shape) or Input(Shape). The
data in Input(X) are unchanged.

Examples:

1. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [6, 8], the reshape operator will transform Input(X)
into a 2-D tensor with shape [6, 8] and leaving Input(X)'s data unchanged.

2. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [2, 3, -1, 2], the reshape operator will transform
Input(X) into a 4-D tensor with shape [2, 3, 4, 2] and leaving Input(X)'s data
unchanged. In this case, one and only dimension of Attr(shape) can be set to -1,
the value of this dimension is inferred from the total element number of
Input(X) and remaining dimensions.

3. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [-1, 0, 3, 2], the reshape operator will transform
Input(X) into a 4-D tensor with shape [2, 4, 3, 2] and leaving Input(X)'s data
unchanged. In this case, besides -1, 0 means the actual dimension value is going
to be copied from the corresponding dimension of Input(X).

Note:

1. One and only one dimension in Attr(shape) can be set -1. In this case,
the actual dimension value will be infered from the total element number of
Input(X) and remaining dimensions.

2. More than one dimensions in Attr(shape) can be set to 0, which means the real
dimension value will be copied from Input(X) at runtime. Note that the index of
0 can not exceed Rank(X). For example, Input(X) is a 3-D tensor with shape
[2, 3, 4], Attr(shape) = [2, 3, 2, 0] is an invalid input.

3. Input(Shape) has a higher priority than Attr(shape) if it is provided, while
Attr(shape) still should be set correctly to gurantee shape inference in
compile-time.

)DOC");
  }
};

class ReshapeGradOp : public framework::OperatorWithKernel {
 public:
  ReshapeGradOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::LoDTensor>("X")->type(),
                                   ctx.device_context());
  }
};

class ReshapeKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    ReshapeFunc(ctx);
  }
};

class ReshapeGradKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto in_dims = d_x->dims();

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopySync(*d_out, ctx.GetPlace(), d_x);
    d_x->Resize(in_dims);
  }
};

// FIXME(zcd): reshape2 adds an intermediate output(XShape) based on reshape,
// the XShape is used to carry the shape and lod of X which will be used in
// reshape_grad, in this way, the framework can reuse the memory of X
// immediately the reshape_op is finished.
// Considering compatibility issues, we could not fix reshape_op
class Reshape2Op : public ReshapeOp {
 public:
  Reshape2Op(const std::string &type, const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : ReshapeOp(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("XShape"),
                   "Output(XShape) of ReshapeOp should not be null.");
    const auto &x_dims = ctx->GetInputDim("X");
    std::vector<int64_t> xshape_dims(x_dims.size() + 1);
    xshape_dims[0] = 0;
    for (int i = 0; i < x_dims.size(); ++i) {
      xshape_dims[i + 1] = x_dims[i];
    }
    ctx->SetOutputDim("XShape", framework::make_ddim(xshape_dims));
    ctx->ShareLoD("X", /*->*/ "XShape");

    ReshapeOp::InferShape(ctx);
  }
};

class Reshape2OpMaker : public ReshapeOpMaker {
 public:
  void Make() override {
    ReshapeOpMaker::Make();
    AddOutput("XShape",
              "XShape is just used to store the shape and lod of X, which will "
              "be used in FlattenGradOp.")
        .AsIntermediate();
  }
};

class Reshape2GradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("reshape2_grad");
    grad_op->SetInput("XShape", Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

class Reshape2GradOp : public framework::OperatorWithKernel {
 public:
  Reshape2GradOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("XShape"), "Input(XShape) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    auto xshape_dims = ctx->GetInputDim("XShape");
    auto x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD("XShape", framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"))->type(),
        ctx.device_context());
  }
};

class ReshapeOpInplaceInToOut : public framework::InplaceOpInference {
 public:
  std::unordered_map<std::string, std::string> operator()(
      const framework::OpDesc &op_desc) const override {
    std::unordered_map<std::string, std::string> inplace_in_to_out = {
        {"X", "Out"},
    };
    return inplace_in_to_out;
  }
};

class ReshapeGradInplaceInToOut : public framework::InplaceOpInference {
 public:
  std::unordered_map<std::string, std::string> operator()(
      const framework::OpDesc &op_desc) const override {
    std::unordered_map<std::string, std::string> inplace_in_to_out = {
        {framework::GradVarName("Out"), framework::GradVarName("X")},
    };
    return inplace_in_to_out;
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(reshape, ops::ReshapeOp, ops::ReshapeOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>,
                  ops::ReshapeOpInplaceInToOut);
REGISTER_OPERATOR(reshape_grad, ops::ReshapeGradOp,
                  ops::ReshapeGradInplaceInToOut);
REGISTER_OP_CPU_KERNEL_FUNCTOR(reshape, float, ops::ReshapeKernel, double,
                               ops::ReshapeKernel, int, ops::ReshapeKernel,
                               int64_t, ops::ReshapeKernel);
REGISTER_OP_CPU_KERNEL_FUNCTOR(reshape_grad, float, ops::ReshapeGradKernel,
                               double, ops::ReshapeGradKernel, int,
                               ops::ReshapeGradKernel, int64_t,
                               ops::ReshapeGradKernel);

REGISTER_OPERATOR(reshape2, ops::Reshape2Op, ops::Reshape2OpMaker,
                  ops::Reshape2GradMaker, ops::ReshapeOpInplaceInToOut);
REGISTER_OPERATOR(reshape2_grad, ops::Reshape2GradOp,
                  ops::ReshapeGradInplaceInToOut);
REGISTER_OP_CPU_KERNEL_FUNCTOR(reshape2, float, ops::ReshapeKernel, double,
                               ops::ReshapeKernel, int, ops::ReshapeKernel,
                               int64_t, ops::ReshapeKernel, int8_t,
                               ops::ReshapeKernel, uint8_t, ops::ReshapeKernel);
REGISTER_OP_CPU_KERNEL_FUNCTOR(reshape2_grad, float, ops::ReshapeGradKernel,
                               double, ops::ReshapeGradKernel, int,
                               ops::ReshapeGradKernel, int64_t,
                               ops::ReshapeGradKernel);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL_FUNCTOR(reshape, float, ops::ReshapeKernel, double,
                                ops::ReshapeKernel, int, ops::ReshapeKernel,
                                int64_t, ops::ReshapeKernel, plat::float16,
                                ops::ReshapeKernel);
REGISTER_OP_CUDA_KERNEL_FUNCTOR(reshape_grad, float, ops::ReshapeGradKernel,
                                double, ops::ReshapeGradKernel, int,
                                ops::ReshapeGradKernel, int64_t,
                                ops::ReshapeGradKernel, plat::float16,
                                ops::ReshapeGradKernel);
REGISTER_OP_CUDA_KERNEL_FUNCTOR(reshape2, float, ops::ReshapeKernel, double,
                                ops::ReshapeKernel, int, ops::ReshapeKernel,
                                int64_t, ops::ReshapeKernel, plat::float16,
                                ops::ReshapeKernel);
REGISTER_OP_CUDA_KERNEL_FUNCTOR(reshape2_grad, float, ops::ReshapeGradKernel,
                                double, ops::ReshapeGradKernel, int,
                                ops::ReshapeGradKernel, int64_t,
                                ops::ReshapeGradKernel, plat::float16,
                                ops::ReshapeGradKernel);
#endif
