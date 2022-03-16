/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

class ViterbiDecodeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class ViterbiDecodeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "The unary emission tensor. The shape of Input must be (batch_size,"
        "sequence_length, num_tags). ");
    AddInput("Transition",
             "The transition matrix. The shape of Transition must be ( "
             "num_tags, num_tags). ");
    AddInput("Length",
             "The input length tensor storing real length of each sequence for "
             "correctness. The shape of Length MUST be (batch_size).");
    AddOutput("Scores",
              "The scores tensor containing the score for the Viterbi "
              "sequence. The shape of Scores MUST be (batch_size).");
    AddOutput("Path",
              "The paths tensor containing the highest scoring tag indices. "
              "The shape of Scores MUST be (batch_size, sequence_length).");
    AddAttr<bool>("include_bos_eos_tag",
                  "If set to True, the last row and the last column of "
                  "transitions will be considered as start tag.")
        .SetDefault(true);
    AddComment(R"DOC(
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace platform = paddle::platform;
DECLARE_INFER_SHAPE_FUNCTOR(viterbi_decode, ViterbiDecodeInferShapeFunctor,
                            PD_INFER_META(phi::ViterbiDecodeInferMeta));
REGISTER_OP_WITHOUT_GRADIENT(viterbi_decode, ops::ViterbiDecodeOp,
                             ops::ViterbiDecodeOpMaker,
                             ViterbiDecodeInferShapeFunctor);
