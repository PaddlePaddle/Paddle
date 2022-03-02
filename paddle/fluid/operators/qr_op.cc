// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/qr_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/phi/core/ddim.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {
using DDim = framework::DDim;

class QrOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

template <typename T>
class QrGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("qr_grad");
    retv->SetInput(framework::GradVarName("Q"), this->OutputGrad("Q"));
    retv->SetInput(framework::GradVarName("R"), this->OutputGrad("R"));
    retv->SetInput("Q", this->Output("Q"));
    retv->SetInput("R", this->Output("R"));
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DELCARE_INFER_SHAPE_FUNCTOR(qr, QrInferShapeFunctor,
                            PT_INFER_META(phi::QrInferMeta));

REGISTER_OPERATOR(qr, ops::QrOp, ops::QrOpMaker,
                  ops::QrGradMaker<paddle::framework::OpDesc>,
                  ops::QrGradMaker<paddle::imperative::OpBase>,
                  QrInferShapeFunctor);

REGISTER_OPERATOR(qr_grad, ops::QrGradOp);

REGISTER_OP_CPU_KERNEL(
    qr_grad, ops::QrGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::QrGradKernel<paddle::platform::CPUDeviceContext, double>);
