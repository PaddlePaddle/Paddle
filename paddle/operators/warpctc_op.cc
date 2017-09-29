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

#include "paddle/operators/warpctc_op.h"

namespace paddle {
namespace operators {

class WarpCTCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {}
};

class WarpCTCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  WarpCTCOpMaker(framework::OpProto* proto,
                 framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Logits", "(LodTensor)");
    AddInput("Label", "(LodTensor)");
    AddInput("Gradient", "(LodTensor)");
    AddOutput("Loss", "(Tensor)");
    AddAttr<int>("blank", "").SetDefault(0);
    AddAttr<bool>("normByTimes", "").SetDefault(false);
    AddComment(R"DOC(
warp-ctc
)DOC");
  }
};

class WarpCTCGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
