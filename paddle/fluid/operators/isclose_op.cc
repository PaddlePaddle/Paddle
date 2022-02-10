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

#include "paddle/fluid/operators/isclose_op.h"
#include <cmath>
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

template <typename T>
struct GetTensorValue<platform::CPUDeviceContext, T> {
  T operator()(const platform::CPUDeviceContext& dev_ctx,
               const framework::Tensor& tensor) const {
    return *(tensor.data<T>());
  }
};

template <typename T>
struct IscloseFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& other,
                  const double rtol, const double atol, bool equal_nan,
                  framework::Tensor* output) {
    auto* in_a = in.data<T>();
    auto* in_b = other.data<T>();
    auto* out_data = output->mutable_data<bool>(ctx.GetPlace());
    auto num = in.numel();
    // *out_data = true;
    for (int i = 0; i < num; i++) {
      out_data[i] = true;
    }
    for (int i = 0; i < num; i++) {
      const T a = in_a[i], b = in_b[i];
      bool val;
      if (std::isnan(a) || std::isnan(b)) {
        val = equal_nan && std::isnan(a) == std::isnan(b);
      } else {
        T left = (a > b ? a - b : b - a);
        T right = atol + (b > 0 ? rtol * b : (-rtol) * b);
        T diff = (left > right ? left - right : right - left);
        val = a == b || left <= right || diff <= 1e-15;
      }
      // *out_data &= val;
      out_data[i] = val;
    }
  }
};

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

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "Isclose");
    OP_INOUT_CHECK(ctx->HasInput("Other"), "Input", "Other", "Isclose");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Isclose");

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

    ctx->SetOutputDim("Out", input_dim);
  }

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
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(
    isclose, ops::IscloseOp, ops::IscloseOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::IscloseOpVarTypeInference);
REGISTER_OP_CPU_KERNEL(isclose, ops::IscloseKernel<CPU, float>,
                       ops::IscloseKernel<CPU, double>);
