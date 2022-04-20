/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class RReluOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

//   void InferShape(framework::InferShapeContext* ctx) const override {
//     OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Dropout");

//     auto x_dims = ctx->GetInputDim("X");
//     ctx->SetOutputDim("Out", x_dims);
//     if (ctx->Attrs().Get<bool>("is_test") == false) {
//       ctx->SetOutputDim("Mask", x_dims);
//     }
//     ctx->ShareLoD("X", /*->*/ "Out");
//   }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }

//   framework::OpKernelType GetKernelTypeForVar(
//       const std::string& var_name, const Tensor& tensor,
//       const framework::OpKernelType& expected_kernel_type) const override {
//     if (var_name == "Seed") {
//       VLOG(10) << "var_name:" << var_name
//                << " does not need to transform in dropout op";
//       return expected_kernel_type;
//     }

//     return framework::OpKernelType(expected_kernel_type.data_type_,
//                                    tensor.place(), tensor.layout());
//   }
};

class RReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of rrelu op.");
    AddInput("Seed",
             "The seed of rrelu op, it has higher priority than the attr "
             "fix_seed and seed")
        .AsDispensable()
        .AsExtra();
    AddOutput("Out", "The output of rrelu op.");
    AddOutput("Mask", "The random sampled rrelu mask.")
        .AsIntermediate()
        .AsExtra();

    // AddAttr<float>("dropout_prob", "Probability of setting units to zero.")
    //     .SetDefault(.5f)
    //     .AddCustomChecker([](const float& drop_p) {
    //       PADDLE_ENFORCE_EQ(drop_p >= 0.0f && drop_p <= 1.0f, true,
    //                         platform::errors::InvalidArgument(
    //                             "'dropout_prob' must be between 0.0 and 1.0."));
    //     });
    AddAttr<float>("lower", "lower bound of the uniform distribution")
        .SetDefault(.125f)
        .AddCustomChecker([](const float& lower) {
          PADDLE_ENFORCE_EQ(lower >= 0.0f && lower <= 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'lower' must be in the range [0.0, 1.0]."));
        });
    AddAttr<float>("upper", "upper bound of the uniform distribution")
        .SetDefault(.33333f)
        .AddCustomChecker([](const float& upper) {
          PADDLE_ENFORCE_EQ(upper >= 0.0f && upper <= 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'upper' must be in the range [0.0, 1.0]."));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<bool>("fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug so that all the negative elements in "
                  "a tensor will be multiplied by the fixed random sampled values.")
        .SetDefault(false)
        .AsExtra();
    AddAttr<int>("seed", "RRelu random seed.").SetDefault(0).AsExtra();
    // AddAttr<std::string>(
    //     "dropout_implementation",
    //     "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
    //     "There are two kinds of ways to implement dropout"
    //     "(the mask below is a tensor have the same shape with input"
    //     "the value of mask is 0 or 1, the ratio of 0 is dropout_prob)"
    //     "1. downgrade_in_infer(default), downgrade the outcome at inference "
    //     "time"
    //     "   train: out = input * mask"
    //     "   inference: out = input * (1.0 - dropout_prob)"
    //     "2. upscale_in_train, upscale the outcome at training time, do nothing "
    //     "in inference"
    //     "   train: out = input * mask / ( 1.0 - dropout_prob )"
    //     "   inference: out = input"
    //     "   dropout op can be removed from the program. the program will be "
    //     "efficient")
    //     .SetDefault("downgrade_in_infer")
    //     .AddCustomChecker([](const std::string& type) {
    //       PADDLE_ENFORCE_EQ(
    //           type == "downgrade_in_infer" || type == "upscale_in_train", true,
    //           platform::errors::InvalidArgument(
    //               "dropout_implementation can only be downgrade_in_infer or "
    //               "upscale_in_train"));
    //     });

    AddComment(R"DOC(
RRelu Operator.

Applies the randomized leaky rectified liner unit function, 
element-wise, as described in the paper:
`Empirical Evaluation of Rectified Activations in Convolutional Network`.

The function is defined as:
.. math::
    \text{RReLU}(x) =
    \begin{cases}
        x & \text{if } x \geq 0 \\
        ax & \text{ otherwise }
    \end{cases}
where :math:`a` is randomly sampled from uniform distribution
:math:`\mathcal{U}(\text{lower}, \text{upper})`.
 See: https://arxiv.org/pdf/1505.00853.pdf

)DOC");
  }
};

class RReluGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

//   void InferShape(framework::InferShapeContext* ctx) const override {
//     OP_INOUT_CHECK(ctx->HasInput("Mask"), "Input", "Mask", "DropoutGrad");
//     OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
//                    framework::GradVarName("Out"), "DropoutGrad");

//     auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

//     ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
//     ctx->ShareLoD(framework::GradVarName("Out"),
//                   /*->*/ framework::GradVarName("X"));
//   }
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Mask"), "Input", "Mask", "rrelu_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "rrelu_grad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"), "rrelu_grad");

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
    ctx->ShareLoD(framework::GradVarName("Out"),
                  /*->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class RReluGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("rrelu_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetInput("Mask", this->Output("Mask"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};
// DECLARE_NO_NEED_BUFFER_VARS_INFERER(TraceGradNoNeedBufferVarsInferer, "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(rrelu, RReluInferShapeFunctor,
                            PD_INFER_META(phi::RReluInferMeta));
REGISTER_OPERATOR(rrelu, ops::RReluOp, ops::RReluOpMaker,
                  ops::RReluGradOpMaker<paddle::framework::OpDesc>,
                  ops::RReluGradOpMaker<paddle::imperative::OpBase>,
                  RReluInferShapeFunctor);
REGISTER_OPERATOR(rrelu_grad, ops::RReluGradOp);

// REGISTER_OPERATOR(trace_grad, ops::TraceGradOp,
//                   ops::TraceGradNoNeedBufferVarsInferer);
