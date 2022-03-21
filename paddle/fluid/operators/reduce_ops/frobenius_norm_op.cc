// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

template <typename T>
class FrobeniusNormOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("frobenius_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

class FrobeniusNormOpMaker : public ops::ReduceOpMaker {
 protected:
  virtual std::string GetName() const { return "frobenius_norm"; }
  virtual std::string GetOpType() const { return "Reduce frobenius_norm"; }
};

DECLARE_INFER_SHAPE_FUNCTOR(frobenius_norm, FrobeniusNormInferShapeFunctor,
                            PD_INFER_META(phi::ReduceInferMetaBase));

REGISTER_OPERATOR(frobenius_norm, ops::ReduceOp, FrobeniusNormOpMaker,
                  ops::FrobeniusNormOpGradMaker<paddle::framework::OpDesc>,
                  ops::FrobeniusNormOpGradMaker<paddle::imperative::OpBase>,
                  FrobeniusNormInferShapeFunctor);

REGISTER_OPERATOR(frobenius_norm_grad, ops::ReduceGradOp);
