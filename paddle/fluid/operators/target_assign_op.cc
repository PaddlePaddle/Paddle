/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/target_assign_op.h"

namespace paddle {
namespace operators {

class TargetAssignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // checkout inputs
    PADDLE_ENFORCE(ctx->HasInput("EncodedGTBBox"),
                   "Input(EncodedGTBBox) of TargetAssignOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput("GTScoreLabel"),
                   "Input(GTScoreLabel) of TargetAssignOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput("MatchIndices"),
                   "Input(MatchIndices) of TargetAssignOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput("NegIndices"),
                   "Input(NegIndices) of TargetAssignOp should not be null");

    // checkout outputs
    PADDLE_ENFORCE(
        ctx->HasOutput("PredBBoxLabel"),
        "Output(PredBBoxLabel) of TargetAssignOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("PredBBoxWeight"),
        "Output(PredBBoxWeight) of TargetAssignOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("PredScoreLabel"),
        "Output(PredScoreLabel) of TargetAssignOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("PredScoreWeight"),
        "Output(PredScoreWeight) of TargetAssignOp should not be null.");

    auto blabel_dims = ctx->GetInputDim("EncodedGTBBox");
    auto slabel_dims = ctx->GetInputDim("GTScoreLabel");
    auto mi_dims = ctx->GetInputDim("MatchIndices");
    auto neg_dims = ctx->GetInputDim("NegIndices");

    PADDLE_ENFORCE_EQ(blabel_dims.size(), 3UL,
                      "The rank of Input(EncodedGTBBox) must be 3.");
    PADDLE_ENFORCE_EQ(slabel_dims.size(), 2UL,
                      "The rank of Input(GTScoreLabel) must be 2.");
    PADDLE_ENFORCE_EQ(mi_dims.size(), 2UL,
                      "The rank of Input(MatchIndices) must be 2.");
    PADDLE_ENFORCE_EQ(neg_dims.size(), 2UL,
                      "The rank of Input(NegIndices) must be 2.");

    PADDLE_ENFORCE_EQ(blabel_dims[0], slabel_dims[0],
                      "The 1st dimension (means the total number of "
                      "ground-truth bounding boxes) of Input(EncodedGTBBox) "
                      "and Input(GTScoreLabel) must be the same.");
    PADDLE_ENFORCE_EQ(blabel_dims[1], mi_dims[1],
                      "The 2nd dimension (means the number of priod boxes) "
                      "of Input(EncodedGTBBox) and "
                      "Input(MatchIndices) must be the same.");
    PADDLE_ENFORCE_EQ(blabel_dims[2], 4,
                      "The 3rd dimension of Input(EncodedGTBBox) must be 4.");

    auto n = mi_dims[0];
    auto np = mi_dims[1];
    ctx->SetOutputDim("PredBBoxLabel", {n, np, 4});
    ctx->SetOutputDim("PredBBoxWeight", {n, np, 1});
    ctx->SetOutputDim("PredScoreLabel", {n, np, 1});
    ctx->SetOutputDim("PredScoreWeight", {n, np, 1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(
            ctx.Input<framework::LoDTensor>("EncodedGTBBox")->type()),
        ctx.device_context());
  }
};

class TargetAssignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TargetAssignOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("EncodedGTBBox",
             "(LoDTensor), The encoded ground-truth bounding boxes with shape "
             "[Ng, Np, 4], where Ng is the total number of ground-truth boxes "
             "in this mini-batch, Np the number of predictions, 4 is the "
             "number of coordinate in [xmin, ymin, xmax, ymax] layout.");
    AddInput("GTScoreLabel",
             "(LoDTensor, default LoDTensor<int>),  The input ground-truth "
             "labels with shape [Ng, 1], where the Ng is the same as it in "
             "the input of EncodedGTBBox.");
    AddInput("MatchIndices",
             "(Tensor, default Tensor<int>), The input matched indices "
             "with shape [N, Np], where N is the batch size, Np is the same "
             "as it in the input of EncodedGTBBox. If MatchIndices[i][j] "
             "is -1, the j-th prior box is not matched to any ground-truh "
             "box in i-th instance.");
    AddInput("NegIndices",
             "(LoDTensor, default LoDTensor<int>), The input negative example "
             "indices with shape [Neg, 1], where is the total number of "
             "negative example indices.");
    AddAttr<int>("background_label",
                 "(int, default 0), Label index of background class.")
        .SetDefault(0);
    AddOutput("PredBBoxLabel",
              "(Tensor), The output encoded ground-truth labels "
              "with shape [N, Np, 4], N is the batch size and Np, 4 is the "
              "same as they in input of EncodedGTBBox. If MatchIndices[i][j] "
              "is -1, the PredBBoxLabel[i][j][:] is the encoded ground-truth "
              "box for background_label in i-th instance.");
    AddOutput("PredBBoxWeight",
              "(Tensor), The weight for PredBBoxLabel with the shape "
              "of [N, Np, 1]");
    AddOutput("PredScoreLabel",
              "(Tensor, default Tensor<int>), The output score labels for "
              "each predictions with shape [N, Np, 1]. If MatchIndices[i][j] "
              "is -1, PredScoreLabel[i][j] = background_label.");
    AddOutput("PredScoreWeight",
              "(Tensor), The weight for PredScoreLabel with the shape "
              "of [N, Np, 1]");
    AddComment(R"DOC(
This operator is, for given the encoded boxes between prior boxes and
ground-truth boxes and ground-truth class labels, to assign classification
and regression targets to each prior box as well as weights to each
prior box. The weights is used to specify which prior box would not contribute
to training loss.

For each instance, the output `PredBBoxLabel`, `PredBBoxWeight`,
`PredScoreLabel` and `PredScoreWeight` are assigned based on `MatchIndices`.
Assumed that the row offset for each instance in `EncodedGTBBox` is called lod,
this operato assigns classification/regression targets by performing the
following steps:

1. Assigning all outpts based on `MatchIndices`:

If id = MatchIndices[i][j] > 0,

    PredBBoxLabel[i][j] = EncodedGTBBox[lod[i] + id][j]
    PredBBoxWeight[i][j] = 1.
    PredScoreLabel[i][j] = GTScoreLabel[lod[i] + id]
    PredScoreWeight[i][j] = 1.

Otherwise, 

    PredBBoxLabel[j][j] = [0., 0., 0., 0.]
    PredBBoxWeight[i][j] = 0.
    PredScoreLabel[i][j] = background_label
    PredScoreWeight[i][j] = 0.

2. Assigning PredScoreWeight based on `NegIndices`:

Assumed that the row offset for each instance in `NegIndices` is caleed neg_lod,
for i-th instance and all ids of NegIndices in this instance:

    PredScoreLabel[i][id] = background_label
    PredScoreWeight[i][id] = 1.0

    )DOC");
  }
};

template <typename T>
struct NegTargetAssignFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const int* neg_indices,
                  const size_t* lod, const int num, const int num_prior_box,
                  const int background_label, int* out_label, T* out_label_wt) {
    for (int i = 0; i < num; ++i) {
      for (size_t j = lod[i]; j < lod[i + 1]; ++j) {
        int id = neg_indices[j];
        out_label[i * num_prior_box + id] = background_label;
        out_label_wt[i * num_prior_box + id] = static_cast<T>(1.0);
      }
    }
  }
};

template struct NegTargetAssignFunctor<platform::CPUDeviceContext, float>;
template struct NegTargetAssignFunctor<platform::CPUDeviceContext, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(target_assign, ops::TargetAssignOp,
                             ops::TargetAssignOpMaker);
REGISTER_OP_CPU_KERNEL(
    target_assign,
    ops::TargetAssignKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TargetAssignKernel<paddle::platform::CPUDeviceContext, double>);
