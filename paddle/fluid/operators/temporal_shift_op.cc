/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/temporal_shift_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class TemporalShiftOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class TemporalShiftOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of temporal shift operator. "
             "This is a 4-D tensor with shape of [N*T, C, H, W] "
             "or [N*T, H, W, C]. "
             "While N is the batch size, T is the temporal segment "
             "number, C is the channel number, H is the height of "
             "features and W is the width of features. "
             "The data type is float32 and float64");
    AddOutput("Out",
              "The output tensor of temporal shift operator. "
              "This is a 4-D tensor in the same shape with Input(X).");

    AddAttr<int>("seg_num",
                 "The temporal segment number, this should be a positive "
                 "integer.");
    AddAttr<float>(
        "shift_ratio",
        "The shift ratio of the channels, the first :attr:`shift_ratio` part "
        "of channels will be shifted by -1 along the temporal dimension, "
        "and the second :attr:`shift_ratio` part of channels will be shifted "
        "by 1 along the temporal dimension. :attr:`shift_ratio` should be in "
        "range [0, 0.5]. Default 0.25.")
        .SetDefault(0.25);
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "an optional string from: \"NHWC\", \"NCHW\". "
        "Specify that the data format of the input and output data is "
        "channel_first or channel_last.")
        .SetDefault("NCHW");

    AddComment(R"DOC(
          This operator calculates the temporal shifting features for Input(X).

          Input(X) should be in shape of [N*T, C, H, W] or [N*T, H, W, C], while 
          N is the batch size, T is the temporal segment number specified by 
          :attr:`seg_num`, C is the channel number, H and W is the height and 
          width of features.

          Temporal Shifting is calculated as follows when data format is NCHW:
          
          Step 1: Reshape Input(X) to [N, T, C, H, W].

          Step 2: Pad 0 to reshaping result in the 2nd(T) dimension with 
          padding width as 1 on each side, padding result will be in shape 
          of [N, T+2, C, H, W].

          Step 3: Assume :attr:`shift_ratio` is :math:`1/4`, slice padding 
          result as follows:

          $$
          slice1 = x[:, :T, :C/4, :, :]
          $$
          $$
          slice2 = x[:, 2:T+2, C/4:C/2, :, :]
          $$
          $$
          slice3 = x[:, 1:T+1, C/2:, :, :]
          $$

          Step 4: Concatenate three slices along the 3rd(C) dimension and 
          reshape result to [N*T, C, H, W].

          For details of temporal shifting, please refer to paper: 
          `Temporal Shift Module <http://arxiv.org/abs/1811.08383>`_ .

         )DOC");
  }
};

class TemporalShiftOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"),
                        ctx->GetInputDim(framework::GradVarName("Out")));
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class TemporalShiftGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("temporal_shift_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(temporal_shift, TemporalShiftInferShapeFunctor,
                            PD_INFER_META(phi::TemporalShiftInferMeta));
REGISTER_OPERATOR(temporal_shift, ops::TemporalShiftOp,
                  ops::TemporalShiftOpMaker,
                  ops::TemporalShiftGradOpMaker<paddle::framework::OpDesc>,
                  ops::TemporalShiftGradOpMaker<paddle::imperative::OpBase>,
                  TemporalShiftInferShapeFunctor);
REGISTER_OPERATOR(temporal_shift_grad, ops::TemporalShiftOpGrad);
REGISTER_OP_CPU_KERNEL(temporal_shift, ops::TemporalShiftKernel<float>,
                       ops::TemporalShiftKernel<double>);
REGISTER_OP_CPU_KERNEL(temporal_shift_grad, ops::TemporalShiftGradKernel<float>,
                       ops::TemporalShiftGradKernel<double>);
