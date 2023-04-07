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
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class BilinearOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class BilinearOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The first input of bilinear operator.");
    AddInput("Y", "The second input of bilinear operator.");
    AddInput("Weight", "The learnable parameters of bilinear operator.");
    AddInput("Bias", "The learnable bias of bilinear operator.")
        .AsDispensable();
    AddOutput("Out", "The output of bilinear operator.");
    AddComment(R"DOC(
Bilinear Tensor Product operator.
Given input X and Y, a 3D tensor Weight and a Bias. Each column of the
Output is computed by one slice $i = 1, . . . , k$ of the tensor:

$$
M =  (X W_i) * Y \\
Out_i = \sum_j {M_j} + Bias_i
$$

Where $W_i$ is the $i$-th slice of Input(Weight);
      $M_j$ is the $j$-th column of $M$;
      $Out_i$ is the $i$-th column of Output(Out);
      $Bias_i$ is a column vector, each element of it is equal to
        the $i$-th element of $Bias$;

)DOC");
  }
};

class BilinearOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class BilinearGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("bilinear_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Weight", this->Input("Weight"));
    if (this->HasInput("Bias")) {
      op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    }

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetOutput(framework::GradVarName("Weight"), this->InputGrad("Weight"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(bilinear,
                            BilinearInferShapeFunctor,
                            PD_INFER_META(phi::BilinearInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(bilinear_grad,
                            BilinearGradInferShapeFunctor,
                            PD_INFER_META(phi::BilinearGradInferMeta));

REGISTER_OPERATOR(bilinear,
                  ops::BilinearOp,
                  ops::BilinearOpMaker,
                  ops::BilinearGradOpMaker<paddle::framework::OpDesc>,
                  ops::BilinearGradOpMaker<paddle::imperative::OpBase>,
                  BilinearInferShapeFunctor);
REGISTER_OPERATOR(bilinear_grad,
                  ops::BilinearOpGrad,
                  BilinearGradInferShapeFunctor);
