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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class UniqueConsecutiveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

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
DECLARE_INFER_SHAPE_FUNCTOR(unique_consecutive,
                            UniqueConsecutiveInferShapeFunctor,
                            PD_INFER_META(phi::UniqueConsecutiveInferMeta));
REGISTER_OP_WITHOUT_GRADIENT(unique_consecutive,
                             ops::UniqueConsecutiveOp,
                             ops::UniqueConsecutiveOpMaker,
                             UniqueConsecutiveInferShapeFunctor);
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
