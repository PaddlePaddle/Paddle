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

#include "paddle/fluid/operators/unique_op.h"
#include <memory>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class UniqueOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "unique");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "unique");

    bool return_index = ctx->Attrs().Get<bool>("return_index");
    bool return_inverse = ctx->Attrs().Get<bool>("return_inverse");
    bool return_counts = ctx->Attrs().Get<bool>("return_counts");
    auto axis_vec = ctx->Attrs().Get<std::vector<int>>("axis");
    auto data_type =
        static_cast<phi::DataType>(static_cast<framework::proto::VarType::Type>(
            ctx->Attrs().Get<int>("dtype")));

    // Construct MetaTensor for InferMeta Func
    using CompatMetaTensor = framework::CompatMetaTensor;
    CompatMetaTensor x(ctx->GetInputVarPtrs("X")[0], ctx->IsRuntime());
    CompatMetaTensor out(ctx->GetOutputVarPtrs("Out")[0], ctx->IsRuntime());
    std::unique_ptr<CompatMetaTensor> indices(nullptr);
    std::unique_ptr<CompatMetaTensor> index(nullptr);
    std::unique_ptr<CompatMetaTensor> counts(nullptr);

    if (return_index) {
      OP_INOUT_CHECK(ctx->HasOutput("Indices"), "Output", "Indices", "unique");
      indices =
          std::move(std::unique_ptr<CompatMetaTensor>(new CompatMetaTensor(
              ctx->GetOutputVarPtrs("Indices")[0], ctx->IsRuntime())));
    }
    if (return_inverse) {
      OP_INOUT_CHECK(ctx->HasOutput("Index"), "Output", "Index", "unique");
      index = std::move(std::unique_ptr<CompatMetaTensor>(new CompatMetaTensor(
          ctx->GetOutputVarPtrs("Index")[0], ctx->IsRuntime())));
    }
    if (return_counts) {
      OP_INOUT_CHECK(ctx->HasOutput("Counts"), "Output", "Counts", "unique");
      counts = std::move(std::unique_ptr<CompatMetaTensor>(new CompatMetaTensor(
          ctx->GetOutputVarPtrs("Counts")[0], ctx->IsRuntime())));
    }
    bool is_sorted = ctx->Attrs().Get<bool>("is_sorted");
    if (is_sorted) {
      phi::UniqueInferMeta(x, return_index, return_inverse, return_counts,
                           axis_vec, data_type, &out, indices.get(),
                           index.get(), counts.get());
    } else {
      OP_INOUT_CHECK(ctx->HasOutput("Index"), "Output", "Index", "unique");
      if (index == nullptr) {
        index =
            std::move(std::unique_ptr<CompatMetaTensor>(new CompatMetaTensor(
                ctx->GetOutputVarPtrs("Index")[0], ctx->IsRuntime())));
      }
      phi::UniqueRawInferMeta(x, return_index, return_inverse, return_counts,
                              axis_vec, data_type, is_sorted, &out,
                              indices.get(), index.get(), counts.get());
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // Return CPUPlace when Attr("is_sorted") is false. Because it means
    // that fluid.layers.unique is called, but there is no cuda kernel.
    if (!ctx.Attr<bool>("is_sorted")) {
      return framework::OpKernelType(
          OperatorWithKernel::IndicateVarDataType(ctx, "X"),
          platform::CPUPlace());
    } else {
      // new version paddle.unique is called.
      return framework::OpKernelType(
          OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
    }
  }
};

class UniqueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Input tensor. It should be a 1-D tensor when Attr(is_sorted)"
             " is fasle or a N-D tensor when Attr(is_sorted) is true.");
    AddAttr<int>("dtype", "data type for output index");
    AddOutput("Out", "A unique subsequence for input tensor.");
    AddOutput("Index",
              "Equivalent to inverse in numpy.unique, "
              "the indices for where elements in the original input ended up "
              "in the returned unique tensor.");
    AddOutput(
        "Indices",
        "The indices of the input tensor that result in the unique tensor.")
        .AsDispensable();
    AddOutput("Counts", "The counts for each unique element.").AsDispensable();
    AddAttr<bool>("return_index",
                  "If True, also return the indices of the input"
                  " tensor that result in the unique Tensor.")
        .SetDefault(false);
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
    AddAttr<bool>("is_sorted",
                  "If True, the unique elements of X are in ascending order."
                  "Otherwise, the unique elements are not sorted.")
        .SetDefault(false);
    AddComment(R"DOC(
    1. Return a unique subsequence for 1-D input tensor, and an index tensor
    pointing to this unique subsequence when Attr(is_sorted) is false. This 
    means paddle.unique is called.
    
    2. Returns the unique elements of X in ascending order when Attr(is_sorted)
    is true. This means fluid.layers.unique is called.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(unique, ops::UniqueOp, ops::UniqueOpMaker);
