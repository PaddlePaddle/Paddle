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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

class EditDistanceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.device_context());
  }
};

class EditDistanceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Hyps",
             "2-D Tensor<int64_t>, or 2-D LoDTensor<int64_t> with last "
             "dimension being 1. "
             "The indices for hypothesis strings.");
    AddInput("Refs",
             "2-D Tensor<int64_t>, or 2-D LoDTensor<int64_t> with last "
             "dimension being 1. "
             "The indices for reference strings.");
    AddInput("HypsLength",
             "1-D Tensor<int64_t>. "
             "Sequence length for hyps when hyps is a tensor")
        .AsDispensable();
    AddInput("RefsLength",
             "1-D Tensor<int64_t>. "
             "Sequence length for refs when refs is a tensor")
        .AsDispensable();
    AddOutput("SequenceNum", "The sequence count of current batch");
    AddAttr<bool>("normalized",
                  "(bool, default false) Indicated whether to normalize "
                  "the edit distance by the length of reference string.")
        .SetDefault(false);
    AddOutput("Out",
              "(2-D Tensor with shape [`batch_size` x 1]) "
              "The output edit distances of EditDistance operator.");
    AddComment(R"DOC(

EditDistance operator computes the edit distances between a batch of hypothesis
strings and their references.

Edit distance, also called Levenshtein distance, measures how dissimilar two strings
are by counting the minimum number of operations to transform one string into another.
The operations include insertion, deletion, and substitution.

For example, given hypothesis string A = "kitten" and reference B = "sitting",
A will be transformed into B at least after two substitutions and one
insertion:

   "kitten" -> "sitten" -> "sittin" -> "sitting"

So the edit distance between A and B is 3.

Input(Hyps) is a 2-D Tensor or a 2-D LoDTensor consisting of all the hypothesis strings.
And the `batch_size` reference strings are arranged in order in the same way in the
Input(Refs).

Output(Out) contains the `batch_size` results and each stands for the edit distance
for a pair of strings respectively. If Attr(normalized) is true, the edit distance
will be divided by the length of reference string.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(edit_distance,
                            EditDistanceShapeFunctor,
                            PD_INFER_META(phi::EditDistanceInferMeta));

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    edit_distance,
    ops::EditDistanceOp,
    ops::EditDistanceOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    EditDistanceShapeFunctor);
