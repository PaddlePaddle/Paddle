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
struct AllcloseFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& other,
                  const double rtol, const double atol, bool equal_nan,
                  framework::Tensor* output) {
    auto* in_a = in.data<T>();
    auto* in_b = other.data<T>();
    auto* out_data = output->mutable_data<bool>(ctx.GetPlace());
    auto num = in.numel();
    *out_data = true;
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
      *out_data &= val;
    }
  }
};

class AllcloseOpMaker : public framework::OpProtoAndCheckerMaker {
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
operator is analogous to :math:`numpy.allclose`, namely that it returns :math:`True` if
two tensors are elementwise equal within a tolerance.
)DOC");
  }
};

class AllcloseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "Allclose");
    OP_INOUT_CHECK(ctx->HasInput("Other"), "Input", "Other", "Allclose");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Allclose");

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

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(allclose)
    .AddCheckpoint(
        R"ROC(Upgrade allclose, add two new inputs [Rtol] and [Atol].)ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewInput("Rtol",
                      "The added input 'Rtol' is not"
                      "dispensable.")
            .NewInput("Atol",
                      "The added input 'Atol' is not"
                      "dispensable."))
    .AddCheckpoint(
        R"ROC(Delete two float attributes [rtol] and [atol], 
        then add 2 string attributes [atol, rtol]. Don't be surprised.
        This is because float cannot represent hight-precision
        floating-point values, and our framework doesn't support
        the use of double attributes. As a result, string instead
        of double is used here to represent high-precision
        floating-point values.
        )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .DeleteAttr("rtol",
                        "The attribute 'rtol' is deleted."
                        "The reason why it is deleted is that"
                        "attributes do not support a float64 value"
                        "and it is changed to a tensor.")
            .DeleteAttr("atol",
                        "The attribute 'atol' is deleted."
                        "The reason why it is deleted is that"
                        "attributes do not support a float64 value"
                        "and it is changed to a tensor.")
            .NewAttr("rtol",
                     "(string) The relative tolerance. Default: :math:`1e-5` .",
                     std::string("1e-5"))
            .NewAttr("atol",
                     "(string) The absolute tolerance. Default: :math:`1e-8` .",
                     std::string("1e-8")));
