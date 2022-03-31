// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

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
namespace operators {
template <typename DeviceContext, typename T, typename Functor>
class OverflowKernel;
}  // namespace operators
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace plat = paddle::platform;

namespace paddle {
namespace operators {

class OverflowV2Op : public framework::OperatorWithKernel {
 public:
  OverflowV2Op(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    int dtype = -1;
    auto *x_var = ctx.InputVar("X");
    if (x_var->IsType<framework::LoDTensor>()) {
      dtype = framework::TransToProtoVarType(
          x_var->Get<framework::LoDTensor>().dtype());
    } else if (x_var->IsType<phi::SelectedRows>()) {
      dtype = framework::TransToProtoVarType(
          x_var->Get<phi::SelectedRows>().value().dtype());
    } else {
      PADDLE_THROW(plat::errors::InvalidArgument(
          "Cannot find the input data type by all input data"));
    }
    return framework::OpKernelType(framework::proto::VarType::Type(dtype),
                                   ctx.GetPlace());
  }
};

class OverflowV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensors of overflowv2 operator.");
    AddOutput("Out",
              "(Tensor) The output tensor of overflowv2 operator. "
              "Same size compare to input tensor");
    AddComment(string::Sprintf(R"DOC(
Overflow %s operator.

$$Out = any(X)$$

Check whether each element of X is Inf or Nan, return the bool result of each
element of X as a tensor.

%s
)DOC",
                               GetName(), GetComments()));
  }

 protected:
  virtual std::string GetName() const = 0;
  virtual std::string GetComments() const = 0;
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(isinf_v2, IsinfInferShapeFunctor,
                            PD_INFER_META(phi::IsfiniteInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(isnan_v2, IsnanInferShapeFunctor,
                            PD_INFER_META(phi::IsfiniteInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(isfinite_v2, IsfiniteInferShapeFunctor,
                            PD_INFER_META(phi::IsfiniteInferMeta));

#define REGISTER_V2OP_MAKER(op_type, comment)           \
  namespace paddle {                                    \
  namespace operators {                                 \
  class _##op_type##OverflowV2OpMaker                   \
      : public ::paddle::operators::OverflowV2OpMaker { \
   protected:                                           \
    std::string GetName() const { return #op_type; }    \
    std::string GetComments() const { return comment; } \
  };                                                    \
  }                                                     \
  }

REGISTER_V2OP_MAKER(isinf_v2, "isinfv2(X)")
REGISTER_V2OP_MAKER(isnan_v2, "isnanv2(X)")
REGISTER_V2OP_MAKER(isfinite_v2, "isfinitev2(X)");

REGISTER_OPERATOR(
    isinf_v2, ops::OverflowV2Op, ops::_isinf_v2OverflowV2OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    IsinfInferShapeFunctor);

REGISTER_OPERATOR(
    isnan_v2, ops::OverflowV2Op, ops::_isnan_v2OverflowV2OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    IsnanInferShapeFunctor);

REGISTER_OPERATOR(
    isfinite_v2, ops::OverflowV2Op, ops::_isfinite_v2OverflowV2OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    IsfiniteInferShapeFunctor);
