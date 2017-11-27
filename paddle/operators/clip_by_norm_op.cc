/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/clip_by_norm_op.h"

namespace paddle {
namespace operators {

class ClipByNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ClipByNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ClipByNormOp should not be null.");
    auto max_norm = ctx->Attrs().Get<float>("max_norm");
    PADDLE_ENFORCE_GT(max_norm, 0, "max_norm should be greater than 0.");
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ClipByNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ClipByNormOpMaker(framework::OpProto* proto,
                    framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor) The input of clip_by_norm op."
             "The number of dimensions must be between [1, 9].");
    AddOutput("Out",
              "(Tensor) The output of clip_by_norm op with shape as input(X)");
    AddAttr<float>("max_norm", "(float) The maximum norm value.");
    AddComment(R"DOC(
ClipByNorm operator limits the L2 norm of the input 'X' within 'max_norm'. 
If the L2 norm of 'X' is less than or equal to 'max_norm', 'Out' will be 
the same as 'X'. If the L2 norm of 'X' is greater than 'max_norm', 'X' will 
be linearly scaled to make the L2 norm of 'Out' equal to 'max_norm', as 
shown in the following formula：

'Out' = 'max_norm' * 'X' / norm('X'),

where norm('X') represents the L2 norm of 'X'.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(clip_by_norm, ops::ClipByNormOp,
                             ops::ClipByNormOpMaker);
REGISTER_OP_CPU_KERNEL(
    clip_by_norm, ops::ClipByNormKernel<paddle::platform::CPUPlace, float>);
