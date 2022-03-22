/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/unique_consecutive_op.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class UniqueConsecutiveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "unique_consecutive");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "unique_consecutive");

    auto in_dims = ctx->GetInputDim("X");
    bool return_inverse = ctx->Attrs().Get<bool>("return_inverse");
    bool return_counts = ctx->Attrs().Get<bool>("return_counts");
    auto axis_vec = ctx->Attrs().Get<std::vector<int>>("axis");
    if (return_inverse) {
      OP_INOUT_CHECK(ctx->HasOutput("Index"), "Output", "Index",
                     "unique_consecutive");
    }
    if (return_counts) {
      OP_INOUT_CHECK(ctx->HasOutput("Counts"), "Output", "Counts",
                     "unique_consecutive");
    }

    if (axis_vec.empty()) {
      ctx->SetOutputDim("Out", {-1});
      if (return_inverse) {
        ctx->SetOutputDim("Index", {phi::product(in_dims)});
      }
    } else {
      int axis = axis_vec[0];
      if (axis < 0) {
        axis += in_dims.size();
      }
      PADDLE_ENFORCE_LT(
          axis, in_dims.size(),
          platform::errors::InvalidArgument("The axis(%d) should be less than "
                                            "the dimension size(%d) of x.",
                                            axis, in_dims.size()));
      auto out_dims = in_dims;
      out_dims[axis] = -1;
      ctx->SetOutputDim("Out", out_dims);
      if (return_inverse) {
        ctx->SetOutputDim("Index", {in_dims[axis]});
      }
    }
    if (return_counts) {
      ctx->SetOutputDim("Counts", {-1});
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class UniqueConsecutiveOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of unique_consecutive op.");
    AddAttr<int>("dtype",
                 "(int, default 5(FP32)) "
                 "data type for output index")
        .SetDefault(framework::proto::VarType::FP32);

    AddOutput("Out", "A unique consecutive subsequence for input tensor.");
    AddOutput("Index",
              "The indices for where elements in the original input ended up "
              "in the returned unique tensor.")
        .AsDispensable();
    AddOutput("Counts", "The counts for each unique element.").AsDispensable();
    AddAttr<bool>(
        "return_inverse",
        "If True, also return the indices for where elements"
        " in the original input ended up in the returned unique tensor.")
        .SetDefault(false);
    AddAttr<bool>("return_counts",
                  "If True, also return the counts for each unique element.")
        .SetDefault(false);
    AddAttr<std::vector<int>>(
        "axis",
        "The axis to apply unique. If None, the input will be flattened.")
        .SetDefault({});
    AddComment(R"DOC(
    This function is different from paddle.unique() in the sense that this
    function only eliminates consecutive duplicate values.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(unique_consecutive, ops::UniqueConsecutiveOp,
                             ops::UniqueConsecutiveOpMaker);
REGISTER_OP_CPU_KERNEL(
    unique_consecutive,
    ops::UniqueConsecutiveKernel<paddle::platform::CPUDeviceContext, float>,
    ops::UniqueConsecutiveKernel<paddle::platform::CPUDeviceContext, double>,
    ops::UniqueConsecutiveKernel<paddle::platform::CPUDeviceContext, int32_t>,
    ops::UniqueConsecutiveKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_VERSION(unique_consecutive)
    .AddCheckpoint(
        R"ROC(
        Upgrade unique_consecutive, add 2 outputs [Indices, Counts] and 3 attribute
        [return_inverse, return_counts, axis].
      )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewOutput("Counts", "The counts for each unique element.")
            .NewAttr("return_inverse",
                     "If True, also return the indices for where elements"
                     " in the original input ended up in the returned unique "
                     "tensor.",
                     false)
            .NewAttr("return_counts",
                     "If True, also return the counts for each unique element.",
                     false)
            .NewAttr("axis",
                     "The axis to apply unique. If None, the input will be "
                     "flattened.",
                     std::vector<int>{}));
