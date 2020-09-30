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

#include "paddle/fluid/operators/allclose_op.h"
#include <cmath>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

template <typename T>
struct AllcloseFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& output, const framework::Tensor& in,
                  const framework::Tensor& other, const float rtol,
                  const float atol, bool equal_nan) {
    auto* in_a = in.data<T>();
    auto* in_b = other.data<T>();
    auto* out_data = output.mutable_data<bool>(ctx.GetPlace());
    auto in_dims = in.numel();
    auto other_dims = other.numel();
    printf("in_dims is :>>>>>>>>>>>>>>>>>>>>>>>>>>>>%ld\n", in_dims);

    PADDLE_ENFORCE_EQ(in_dims == other_dims, true,
                      platform::errors::InvalidArgument(
                          "Dims of input(a) and dims of other(b) should"
                          "be equal, but received the dims of input is : %d ,"
                          "received the dims of other is :%d. ",
                          in_dims, other_dims));

    for (int i = 0; i < in_dims; i++) {
      const T a = in_a[i], b = in_b[i];
      bool val;
      T dif;
      double threshold = 1e-7;
      if (std::isnan(a) || std::isnan(b)) {
        val = equal_nan && isnan(a) == isnan(b);
      } else {
        dif = fabs(fabs(a - b) - (atol + rtol * fabs(b)));
        T left = (a > b ? a - b : b - a);
        T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
        printf("dif is>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: %.15f\n",
               dif);
        printf("left is>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: %.15f\n",
               left);
        printf("right is>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: %.15f\n",
               right);
        printf("rtol is>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: %.15f\n",
               rtol);
        val = a == b ||
              (a > b ? a - b : b - a) <=
                  atol + (b > 0 ? rtol * b : (-rtol) * b) ||
              dif < threshold;
      }
      out_data[i] = val;
      printf("val is>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: %d\n", val);
      printf("in functor out_data[i]: %d\n", out_data[i]);
    }
  }
};

template struct AllcloseFunctor<platform::CPUDeviceContext, float>;

class AllcloseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "The input tensor, it's data type should be float32, float64.");
    AddInput("Other",
             "The input tensor, it's data type should be float32, float64.");
    AddOutput("Out", "The output tensor, it's data type is bool.");

    AddAttr<float>("rtol", "The relative tolerance. Default: :math:`1e-5` .")
        .SetDefault(1e-5);
    AddAttr<float>("atol", "The absolute tolerance. Default: :math:`1e-8` .")
        .SetDefault(1e-8);
    AddAttr<bool>("equal_nan",
                  "If :math:`True` , then two :math:`NaNs` will be "
                  "compared as equal. Default: :math:`False` .")
        .SetDefault(false);

    AddComment(R"DOC( 
This operator checks if all :math:`x` and :math:`y` satisfy the condition:

.. math::
    \left| x - y \right| \leq atol + rtol \times \left| y \right|

elementwise, for all elements of :math:`x` and :math:`y`. The behaviour of this
operator is analogous to :math:`numpy.allclose`, namely that it returns :math:`True` if
two tensors are elementwise equal within a tolerance.
)DOC");
  }
};

class AllcloseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      platform::errors::NotFound(
                          "Input(Input) of allclose op should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Other"), true,
                      platform::errors::NotFound(
                          "Input(Other) of allclose op should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "The output(Out) of allclose op must not be null."));

    auto input_dim = ctx->GetInputDim("Input");
    auto other_dim = ctx->GetInputDim("Other");
    PADDLE_ENFORCE_EQ(input_dim.size(), other_dim.size(),
                      platform::errors::PreconditionNotMet(
                          "Input(Input) and Input(Other) must have the same "
                          "dimension size."));
    int n = input_dim.size();
    bool is_runtime = ctx->IsRuntime();
    for (int i = 0; i < n; i++) {
      if (is_runtime) {
        PADDLE_ENFORCE_EQ(input_dim[i], other_dim[i],
                          platform::errors::PreconditionNotMet(
                              "The value at dim %d of Input(Input) is not "
                              "equal to the Input(Other): %ld != %ld.",
                              i, input_dim[i], other_dim[i]));
      } else {
        if (!(input_dim[i] < 0 || other_dim[i] < 0)) {
          PADDLE_ENFORCE_EQ(input_dim[i], other_dim[i],
                            platform::errors::PreconditionNotMet(
                                "The value at dim %d of Input(Input) is not "
                                "equal to the Input(Other): %ld != %ld.",
                                i, input_dim[i], other_dim[i]));
        }
      }
    }

    ctx->SetOutputDim("Out", framework::make_ddim({1}));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class AllcloseOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    ctx->SetOutputDataType("Out", framework::proto::VarType::BOOL);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(
    allclose, ops::AllcloseOp, ops::AllcloseOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::AllcloseOpVarTypeInference);
REGISTER_OP_CPU_KERNEL(allclose, ops::AllcloseKernel<CPU, float>,
                       ops::AllcloseKernel<CPU, double>);
