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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class RmspropOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class RmspropOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(Tensor, default Tensor<float>) "
             "Input parameter value that has to be updated.");
    AddInput("MeanSquare",
             "(Tensor, default Tensor<float>)"
             " The mean square value that gets updated.");
    AddInput("MeanGrad",
             "(Tensor, default Tensor<float>)"
             " The moving average of gradient")
        .AsDispensable();
    AddInput("LearningRate",
             "(Tensor, default Tensor<float>) "
             "The learning rate should be a tensor of size 1.");
    AddInput("Grad",
             "(Tensor, default Tensor<float>) "
             "Input gradient of the parameter.");
    AddInput("Moment",
             "(Tensor, default Tensor<float>) The moment that gets updated.");

    AddOutput("ParamOut", "(Tensor) Output updated parameter value.");
    AddOutput("MomentOut", "(Tensor) Output updated moment.");
    AddOutput("MeanSquareOut", "(Tensor) Output Mean squared updated value.");
    AddOutput("MeanGradOut",
              "(Tensor) Output moving average of gradient updated value.");

    AddAttr<float>("epsilon",
                   "(float, default 1e-10) Constant "
                   "for numerical stability.")
        .SetDefault(1.0e-10f);
    AddAttr<float>("decay",
                   "(float, default 0.9) "
                   "Discounting factor for coming gradient.")
        .SetDefault(0.9f);
    AddAttr<float>("momentum", "(float, default 0.0) Constant value.")
        .SetDefault(0.0f);
    AddAttr<bool>("centered", "(bool, default false) use centered rmsprop.")
        .SetDefault(false);
    AddComment(R"DOC(
Rmsprop Optimizer.

$$
MeanSquareOut = decay * MeanSquare + (1 - decay) * Grad * Grad \\
MomentOut = momentum * Moment +
            \frac{LearningRate * Grad}{\sqrt{MeanSquareOut + epsilon}} \\
ParamOut = Param -  MomentOut
$$

if centered is true:

mean_grad = decay * mean_square{t-1} + (1-decay) * gradient
mean_square = decay * mean_square{t-1} + (1-decay) * gradient ** 2
mom = momentum * mom{t-1} + learning_rate * g_t /
    sqrt(mean_square - mean_grad**2 + epsilon)
param -= mom

The original slides that proposed Rmsprop: Slide 29 of
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(rmsprop,
                            RmspropInferShapeFunctor,
                            PD_INFER_META(phi::RmspropInferMeta));
REGISTER_OP_WITHOUT_GRADIENT(rmsprop,
                             ops::RmspropOp,
                             ops::RmspropOpMaker,
                             RmspropInferShapeFunctor);
