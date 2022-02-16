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

#include <string>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/pten/core/infermeta_utils.h"
#include "paddle/pten/infermeta/unary.h"

namespace paddle {
namespace operators {

class ScaleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class ScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of scale operator.");
    AddInput("ScaleTensor",
             "(Tensor) If provided, use this as "
             "scale factor, this has a higher priority than "
             "attr(scale), the shape of this tensor MUST BE 1.")
        .AsDispensable();
    AddOutput("Out", "(Tensor) Output tensor of scale operator.");
    AddComment(R"DOC(
**Scale operator**

Apply scaling and bias addition to the input tensor.

if bias_after_scale=True:

$$Out = scale*X + bias$$

else:

$$Out = scale*(X + bias)$$
)DOC");
    AddAttr<float>("scale", "The scaling factor of the scale operator.")
        .SetDefault(1.0);
    AddAttr<float>("bias", "The bias of the scale operator.").SetDefault(0.0);
    AddAttr<bool>(
        "bias_after_scale",
        "Apply bias addition after or before scaling. It is useful for "
        "numeric stability in some circumstances.")
        .SetDefault(true);
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
  }
};

class ScaleOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};

template <typename T>
class ScaleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("scale");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    if (this->HasInput("ScaleTensor") > 0) {
      grad_op->SetInput("ScaleTensor", this->Input("ScaleTensor"));
    }
    grad_op->SetOutput("Out", this->InputGrad("X"));
    VLOG(6) << "Finish SetOutput";
    grad_op->SetAttr("scale", this->GetAttr("scale"));
    VLOG(6) << "Finish Set Attr scale";
    grad_op->SetAttr("bias", 0.0f);
    VLOG(6) << "Finish Set Attr bias";
    grad_op->SetAttr("bias_after_scale", true);
    VLOG(6) << "Finish Set Attr bias_after_scale";
    if (grad_op->HasAttr("use_mkldnn")) {
      VLOG(6) << "Finish Check Attr use_mkldnn";
      grad_op->SetAttr("use_mkldnn", this->GetAttr("use_mkldnn"));
      VLOG(6) << "Finish Set Attr use_mkldnn";
    }
    VLOG(6) << "Finish Apply";
  }
};

DECLARE_INPLACE_OP_INFERER(ScaleOpInplaceInferer, {"X", "Out"});
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DELCARE_INFER_SHAPE_FUNCTOR(scale, ScaleInferShapeFunctor,
                            PT_INFER_META(pten::UnchangedInferMeta));
REGISTER_OPERATOR(scale, ops::ScaleOp, ops::ScaleOpMaker,
                  ops::ScaleGradMaker<paddle::framework::OpDesc>,
                  ops::ScaleGradMaker<paddle::imperative::OpBase>,
                  ScaleInferShapeFunctor, ops::ScaleOpVarTypeInference,
                  ops::ScaleOpInplaceInferer);
