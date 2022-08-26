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

#include "paddle/phi/core/enforce.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class FrameOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

class FrameOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of frame op.");
    AddOutput("Out", "(Tensor), The output tensor of frame op.");
    AddAttr<int>(
        "frame_length",
        "Length of the frame and `0 < frame_length <= x.shape[axis]`.");
    AddAttr<int>("hop_length",
                 "Number of steps to advance between adjacent frames and "
                 "`0 < hop_length`.");
    AddAttr<int>("axis",
                 "Specify the axis to operate on the input Tensors. Its value "
                 "should be 0(the first dimension) or -1(the last dimension).")
        .SetDefault(-1);
    AddComment(R"DOC(
      Slice the N-dimensional (where N >= 1) input into (overlapping) frames.
    )DOC");
  }
};

class FrameOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const auto in_dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(in_dtype, ctx.GetPlace());
  }
};

template <typename T>
class FrameOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("frame_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(frame,
                            FrameInferShapeFunctor,
                            PD_INFER_META(phi::FrameInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(frame_grad,
                            FrameGradInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(frame,
                  ops::FrameOp,
                  ops::FrameOpMaker,
                  ops::FrameOpGradMaker<paddle::framework::OpDesc>,
                  ops::FrameOpGradMaker<paddle::imperative::OpBase>,
                  FrameInferShapeFunctor);

REGISTER_OPERATOR(frame_grad, ops::FrameOpGrad, FrameGradInferShapeFunctor);
