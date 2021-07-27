// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/piano_jit_op.h"

namespace paddle {
namespace operators {

class PianoJitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(PianoJitOp::kInputs, " Input LoDTensorArray of PianoJitOp.");
    AddOutput(PianoJitOp::kOutputs, "Output LoDTensorArray of PianoJitOp.");
    AddOutput(PianoJitOp::kScope, "Output Scope of PianoJitOp.");
    AddAttr<framework::Graph*>(PianoJitOp::kSubGraph,
                               "The SubGraph that PianoJitOp holds.");
    AddComment(R"DOC(
PianoJit Operator.

This operator is used to replace a part of the graph and call the piano compiler to generate the corresponding code.

)DOC");
  }
};

class PianoJitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  static constexpr char kInputs[] = "Inputs";
  static constexpr char kOutputs[] = "Outputs";
  static constexpr char kScope[] = "Scope";
  static constexpr char kSubGraph[] = "SubGraph";

  void InferShape(framework::InferShapeContext* ctx) const override {
    // TODO(levi): need complete.
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(piano_jit, ops::PianoJitOp, ops::PianoJitOpMaker)

REGISTER_OP_CPU_KERNEL(
    piano_jit, ops::PianoJitKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PianoJitKernel<paddle::platform::CPUDeviceContext, double>)
