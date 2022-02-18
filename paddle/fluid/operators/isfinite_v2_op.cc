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

#include "paddle/fluid/operators/isfinite_v2_op.h"

#include <string>

#include "paddle/fluid/operators/common_infer_shape_functions.h"

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
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "isfinitev2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "isfinitev2");
    UnaryOpUnchangedInferShape(ctx);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    int dtype = -1;
    auto *x_var = ctx.InputVar("X");
    if (x_var->IsType<framework::LoDTensor>()) {
      dtype = framework::TransToProtoVarType(
          x_var->Get<framework::LoDTensor>().dtype());
    } else if (x_var->IsType<pten::SelectedRows>()) {
      dtype = framework::TransToProtoVarType(
          x_var->Get<pten::SelectedRows>().value().dtype());
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
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    isnan_v2, ops::OverflowV2Op, ops::_isnan_v2OverflowV2OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    isfinite_v2, ops::OverflowV2Op, ops::_isfinite_v2OverflowV2OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(isnan_v2,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           int, ops::NANV2Functor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           int64_t, ops::NANV2Functor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           float, ops::NANV2Functor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           double, ops::NANV2Functor>,
                       ops::OverflowKernel<paddle::platform::CPUDeviceContext,
                                           plat::float16, ops::NANV2Functor>);

REGISTER_OP_CPU_KERNEL(
    isinf_v2, ops::OverflowKernel<paddle::platform::CPUDeviceContext, int,
                                  ops::InfinityV2Functor>,
    ops::OverflowKernel<paddle::platform::CPUDeviceContext, int64_t,
                        ops::InfinityV2Functor>,
    ops::OverflowKernel<paddle::platform::CPUDeviceContext, float,
                        ops::InfinityV2Functor>,
    ops::OverflowKernel<paddle::platform::CPUDeviceContext, double,
                        ops::InfinityV2Functor>,
    ops::OverflowKernel<paddle::platform::CPUDeviceContext, plat::float16,
                        ops::InfinityV2Functor>);

REGISTER_OP_CPU_KERNEL(
    isfinite_v2, ops::OverflowKernel<paddle::platform::CPUDeviceContext, int,
                                     ops::IsfiniteV2Functor>,
    ops::OverflowKernel<paddle::platform::CPUDeviceContext, int64_t,
                        ops::IsfiniteV2Functor>,
    ops::OverflowKernel<paddle::platform::CPUDeviceContext, float,
                        ops::IsfiniteV2Functor>,
    ops::OverflowKernel<paddle::platform::CPUDeviceContext, double,
                        ops::IsfiniteV2Functor>,
    ops::OverflowKernel<paddle::platform::CPUDeviceContext, plat::float16,
                        ops::IsfiniteV2Functor>);
