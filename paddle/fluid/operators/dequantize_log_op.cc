/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/op_registry.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class DequantizeLogOp : public framework::OperatorWithKernel {
 public:
  DequantizeLogOp(const std::string& type,
                  const framework::VariableNameMap& inputs,
                  const framework::VariableNameMap& outputs,
                  const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return phi::KernelKey(data_type, ctx.device_context().GetPlace());
  }
};

class DequantizeLogOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(int8 Tensor) The input with int8 type is the "
             "low precision tensor.");
    AddInput("Dict", "(float) The Dict in quantization stage.");
    AddOutput("Out",
              "(float32 Tensor) The output is the dequantized high "
              "precision tensor.");
    AddComment(R"DOC(
DequantizeLogOp operator.
This calculation is an opposite operation of QuantizeLogOp:
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(dequantize_log,
                            DequantizeLogInferShapeFunctor,
                            PD_INFER_META(phi::DequantizeLogInferMeta));

REGISTER_OPERATOR(
    dequantize_log,
    ops::DequantizeLogOp,
    ops::DequantizeLogOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    DequantizeLogInferShapeFunctor);
