//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/sequence_ops/sequence_enumerate_op.h"

namespace paddle {
namespace operators {

class SequenceEnumerateOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("X"),
        "Input(X) of SequecceEnumerate operator should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(X) of SequenceEnumerate operator should not be null.");

    const auto x_dims = ctx->GetInputDim("X");
    const auto win_size = ctx->Attrs().Get<int>("win_size");
    // when need_pad is false, shape and lod of output 
    // will be computed in compute kernel
    ctx->SetOutputDim("Out", {x_dims[0], win_size});
    ctx->ShareLoD("X", "Out");
  }
};

class SequenceEnumerateOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(2-D LoDTensor with the 2nd dimension equal to 1) "
             "Input LoDTensor of SequenceEnumerate operator.");
    AddOutput("Out",
              "(2-D LoDTensor with the 2nd dimension equal to win_size) "
              "Output LoDTensor of SequenceEnumerate operator.");
    AddAttr<int>("win_size", "(int) The enumerate sequence window size.")
        .AddCustomChecker([](const int& win_size) {
          PADDLE_ENFORCE(win_size >= 2,
                         "The window size should be not less than 2.");
        });
    AddAttr<int>("pad_value", "(int) The enumerate sequence padding value.")
        .SetDefault(0);
    AddAttr<bool>("need_pad", "(bool) Whether the enumerate sequence needs to pad value.");
    AddAttr<bool>(framework::kAllKernelsMustComputeRuntimeShape,
                  "Skip calling InferShape() function in the runtime.")
        .SetDefault(true);
    AddComment(R"DOC(
Sequence Enumerate Operator.

Generate a new sequence for the input index sequence, which enumerates all the
sub-sequences with length `win_size` of the input. 
The enumerated sequence has the same 1st dimension with variable `input`, and
the 2nd dimension is `win_size`, padded by `pad_value` if necessary in generation.
    
Examples:
Case 1:
  Input:
    X.lod = [[0, 3, 5]]
    X.data = [[1], [2], [3], [4], [5]]
    X.dims = [5, 1]
  Attrs:
    win_size = 2
    pad_value = 0
    need_pad = true
  Output:
    Out.lod = [[0, 3, 5]]
    Out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]
    Out.dims = [5, 2]

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(sequence_enumerate, ops::SequenceEnumerateOp,
                             ops::SequenceEnumerateOpMaker);
REGISTER_OP_CPU_KERNEL(
    sequence_enumerate,
    ops::SequenceEnumerateKernel<paddle::platform::CPUDeviceContext, int32_t>,
    ops::SequenceEnumerateKernel<paddle::platform::CPUDeviceContext, int64_t>);
