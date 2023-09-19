/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fusion_transpose_flatten_concat_op.h"

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class TransposeFlattenConcatFusionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GE(
        ctx->Inputs("X").size(),
        1UL,
        platform::errors::InvalidArgument(
            "Inputs(X) of TransposeFlattenConcat op should not be empty."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"),
        true,
        platform::errors::InvalidArgument(
            "Inputs(X) of TransposeFlattenConcat op should not be empty."));

    auto ins = ctx->GetInputsDim("X");
    const size_t n = ins.size();
    PADDLE_ENFORCE_GT(n,
                      0,
                      platform::errors::InvalidArgument(
                          "The size of Inputs(X)'s dimension should be greater "
                          " than 0, but received %d.",
                          n));

    std::vector<int> trans_axis =
        ctx->Attrs().Get<std::vector<int>>("trans_axis");
    int flatten_axis = ctx->Attrs().Get<int>("flatten_axis");
    int concat_axis = ctx->Attrs().Get<int>("concat_axis");

    size_t x_rank = ins[0].size();
    size_t trans_axis_size = trans_axis.size();
    PADDLE_ENFORCE_EQ(x_rank,
                      trans_axis_size,
                      platform::errors::InvalidArgument(
                          "The input tensor's rank(%d) "
                          "should be equal to the permutation axis's size(%d)",
                          x_rank,
                          trans_axis_size));

    auto dims0 =
        GetFlattenShape(flatten_axis, GetPermuteShape(trans_axis, ins[0]));
    std::vector<int> out_dims(dims0);
    for (size_t i = 1; i < n; i++) {
      auto dimsi =
          GetFlattenShape(flatten_axis, GetPermuteShape(trans_axis, ins[i]));
      for (int j = 0; j < static_cast<int>(dims0.size()); j++) {
        if (j == concat_axis) {
          out_dims[concat_axis] += dimsi[j];
        } else {
          PADDLE_ENFORCE_EQ(out_dims[j],
                            dimsi[j],
                            platform::errors::InvalidArgument(
                                "After flatting, the %d-th dim should be save "
                                "except the specify axis.",
                                j));
        }
      }
    }
    if (out_dims[concat_axis] < 0) {
      out_dims[concat_axis] = -1;
    }
    ctx->SetOutputDim("Out", phi::make_ddim(out_dims));
  }
};

class TransposeFlattenConcatFusionOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor, tensors with rank up to 6 are supported.")
        .AsDuplicable();
    AddOutput("Out", "(Tensor)The output tensor.");
    AddAttr<std::vector<int>>(
        "trans_axis",
        "(vector<int>) A list of values, and the size of the list should be "
        "the same with the input tensor rank. This operator permutes the input "
        "tensor's axes according to the values given.");
    AddAttr<int>("flatten_axis",
                 "(int)"
                 "Indicate up to which input dimensions (exclusive) should be"
                 "flattened to the outer dimension of the output. The value"
                 "for axis must be in the range [0, R], where R is the rank of"
                 "the input tensor. When axis = 0, the shape of the output"
                 "tensor is (1, (d_0 X d_1 ... d_n), where the shape of the"
                 "input tensor is (d_0, d_1, ... d_n).");
    AddAttr<int>("concat_axis",
                 "The axis along which the input tensors will be concatenated. "
                 "It should be 0 or 1, since the tensor is 2D after flatting.");
    AddComment(R"DOC(


)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fusion_transpose_flatten_concat,
    ops::TransposeFlattenConcatFusionOp,
    ops::TransposeFlattenConcatFusionOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
