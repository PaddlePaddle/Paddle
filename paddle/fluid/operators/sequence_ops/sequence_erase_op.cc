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

#include "paddle/fluid/operators/sequence_ops/sequence_erase_op.h"

#include <vector>

namespace paddle {
namespace operators {

class SequenceEraseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SequenceErase");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SequenceErase");
    auto x_dims = ctx->GetInputDim("X");
<<<<<<< HEAD
    PADDLE_ENFORCE(
        x_dims.size() == 2 && x_dims[1] == 1,
        platform::errors::InvalidArgument(
            "Input(X) of SequenceEraseOp should be a 2-D phi::DenseTensor "
            "with the 2nd dimension equal to 1,"
            "but received size %d with the 2nd dimension %d.",
            x_dims.size(),
            x_dims[1]));
    ctx->SetOutputDim("Out", x_dims);
    // The output phi::DenseTensor's lod_level should be input X's lod_level.
    // For compile-time, we call SetLoDLevel to set output's lod_level.
    // For runtime, output phi::DenseTensor's lod is determined by input X's lod
    // and the level specified by input RandTable. We cannot get X's detail lod
    // and RankTable's level in this function, so leave this work to the detail
    // kernel implementation.
=======
    PADDLE_ENFORCE(x_dims.size() == 2 && x_dims[1] == 1,
                   platform::errors::InvalidArgument(
                       "Input(X) of SequenceEraseOp should be a 2-D LoDTensor "
                       "with the 2nd dimension equal to 1,"
                       "but received size %d with the 2nd dimension %d.",
                       x_dims.size(),
                       x_dims[1]));
    ctx->SetOutputDim("Out", x_dims);
    // The output LoDTensor's lod_level should be input X's lod_level.
    // For compile-time, we call SetLoDLevel to set output's lod_level.
    // For runtime, output LoDTensor's lod is determined by input X's lod and
    // the level specified by input RandTable.
    // We cannot get X's detail lod and RankTable's level in this function, so
    // leave this work to the detail kernel implementation.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if (!ctx->IsRuntime()) {
      ctx->SetLoDLevel("Out", ctx->GetLoDLevel("X"));
    }
  }
};

class SequenceEraseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
<<<<<<< HEAD
             "(2-D phi::DenseTensor with the 2nd dim. equal to 1) "
             "Input phi::DenseTensor of SequenceEraseOp.");
    AddOutput("Out",
              "(2-D phi::DenseTensor with the 2nd dim. equal to 1) "
              "Output phi::DenseTensor of SequenceEraseOp.");
=======
             "(2-D LoDTensor with the 2nd dim. equal to 1) "
             "Input LoDTensor of SequenceEraseOp.");
    AddOutput("Out",
              "(2-D LoDTensor with the 2nd dim. equal to 1) "
              "Output LoDTensor of SequenceEraseOp.");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    AddAttr<std::vector<int>>("tokens",
                              "(vector<int>) Tokens need to be erased from "
                              "input sequences.");
    AddComment(R"DOC(
Sequence Erase Operator.

<<<<<<< HEAD
Sequence erase operator erases tokens specified by Attr(tokens) from the input
sequences Input(X), and outputs the remaining data and modifies the LoD
information at the same time. For example, given a 2-D phi::DenseTensor
=======
Sequence erase operator erases tokens specified by Attr(tokens) from the input 
sequences Input(X), and outputs the remaining data and modifies the LoD 
information at the same time. For example, given a 2-D LoDTensor
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    X = [[2, 2, 6, 1, 3, 9, 6, 1, 0, 1]]^T

with lod = [[0, 3, 6, 10]], there are three sequences in the input:
<<<<<<< HEAD

     X1 = [[2, 2, 6]]^T, X2 = [[1, 3, 9]]^T and X3 = [[6, 1, 0, 1]]^T.

If the tokens to be erased are Attr(tokens) = [2, 3, 5], after the erasing
=======
   
     X1 = [[2, 2, 6]]^T, X2 = [[1, 3, 9]]^T and X3 = [[6, 1, 0, 1]]^T.

If the tokens to be erased are Attr(tokens) = [2, 3, 5], after the erasing 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
operation, the three sequences become

    X1' = [[6]]^T, X2' = [[1, 9]]^T and X3' = [[6, 1, 0, 1]]^T.

<<<<<<< HEAD
Hence the phi::DenseTensor Output(Out) should be
=======
Hence the LoDTensor Output(Out) should be
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    Out = [[6, 1, 9, 6, 1, 0, 1]]^T,

with lod = [[0, 1, 3, 7]].

<<<<<<< HEAD
An example usage for this operator is to remove the special tokens when
computing the edit distance between two strings, such as blank, start token,
=======
An example usage for this operator is to remove the special tokens when 
computing the edit distance between two strings, such as blank, start token, 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
and end token.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(sequence_erase,
                             ops::SequenceEraseOp,
                             ops::SequenceEraseOpMaker);
REGISTER_OP_CPU_KERNEL(sequence_erase,
                       ops::SequenceEraseKernel<phi::CPUContext, int32_t>,
                       ops::SequenceEraseKernel<phi::CPUContext, int64_t>);
