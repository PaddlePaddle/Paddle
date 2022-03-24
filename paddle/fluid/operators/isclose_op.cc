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

#include <cmath>
#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class IscloseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "The input tensor, it's data type should be float32, float64.");
    AddInput("Other",
             "The input tensor, it's data type should be float32, float64.");
    AddInput("Rtol", "The relative tolerance.").AsDispensable();
    AddInput("Atol", "The absolute tolerance.").AsDispensable();
    AddOutput("Out", "The output tensor, it's data type is bool.");
    AddAttr<std::string>("rtol",
                         "The relative tolerance. Default: :math:`1e-5` .")
        .SetDefault("1e-5");
    AddAttr<std::string>("atol",
                         "The absolute tolerance. Default: :math:`1e-8` .")
        .SetDefault("1e-8");
    AddAttr<bool>("equal_nan",
                  "If :math:`True` , then two :math:`NaNs` will be "
                  "compared as equal. Default: :math:`False` .")
        .SetDefault(false);

    AddComment(R"DOC( 
This operator checks if all :math:`x` and :math:`y` satisfy the condition:

.. math::
    \left| x - y \right| \leq atol + rtol \times \left| y \right|

elementwise, for all elements of :math:`x` and :math:`y`. The behaviour of this
operator is analogous to :math:`numpy.isclose`, namely that it returns :math:`True` if
two tensors are elementwise equal within a tolerance.
)DOC");
  }
};

class IscloseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class IscloseOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    ctx->SetOutputDataType("Out", framework::proto::VarType::BOOL);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(isclose, IscloseInferShapeFunctor,
                            PD_INFER_META(phi::ValueCompareInferMeta));
REGISTER_OPERATOR(
    isclose, ops::IscloseOp, ops::IscloseOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::IscloseOpVarTypeInference, IscloseInferShapeFunctor);
