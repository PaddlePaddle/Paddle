/*Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/masked_select_op.h"
#include <memory>

namespace paddle {
namespace operators {

class MaskedSelectOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("input"),
                   "Input(input) of MaskedSelectOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("mask"),
                   "Input(mask) of MaskedSelectOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MaskedSelectOp should not be null.");

    auto input_dims = ctx->GetInputDim("input");
    auto mask_dims = ctx->GetInputDim("mask");

    PADDLE_ENFORCE(input_dims.size() == mask_dims.size(),
                   "The input size and the mask size must be equal.")
    for (size_t i = 0; i < input_dims.size(); i++) {
      PADDLE_ENFORCE(input_dims[i] == mask_dims[i],
                     "The input shape and the mask shape must be equal.")
    }
    std::vector<int64_t> out_dims(2);
    out_dims[0] = input_dim[0] int d = 0;
    if (!ctx->IsRunTime()) {
      d = 1;
    } else {
      framework::Variable* mask_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("mask"));
      mask_var.flatten_to_1d(mask_var->numel()) auto* mask_data =
          mask_var->data<bool>()
          // batch size?
          for (size_t i = 0; i < mask_var->numel(); i++) {
        if
          mask_data[i] == 1 : d++;
      }
    }
    out_dims[1] = d ctx->SetOutputDim("Out", framework::make_ddim(out_dims))

    /*
    int input_dims_size = input_dims.size();
    int mask_dims_size = mask_dims.size();

    int output_dims_size = (input_dims_size >= mask_dims_size) ? input_dims_size
    : mask_dims_size;
    //int broad_input_dims[output_dims_size];
    //int broad_mask_dims[output_dims_size];
    //int output_dims[output_dims_size]
    std::vector<int64_t> broad_input_dims(output_dims_size);
    std::vector<int64_t> broad_mask_dims(output_dims_size);
    std::vector<int64_t> output_dims(output_dims_size);

    for (size_t i=0; i< output_dims_size; i++){
            if(i<output_dims_size-input_dims_size){
                    broad_input_dims[i] = 1;
            }else{
                    broad_input_dims[i] = input_dims[i]
            }
    }

    for (size_t i=0; i<output_dims_size; i++){
            if(i<output_dims_size-mask_dims_size){
                    broad_mask_dims[i] = 1;
            }else{
                    broad_input_dims[i] = mask_dims[i]
            }
    }

    for (size_t i=output_dims_size; i>0; i--){
            PADDLE_ENFORCE(broad_input_dims[i]==broad_mask_dims[i] ||
                            broad_input_dims[i]==1|| broad_mask_dims[i]==1,
                            "The input and mask tensor must be braodcastable.")
    }
    for (size_t i=0; i<output_dims_size; i++){
            output_dims[i] = (broad_input_dims[i]>=broad_mask_dims[i]) ?
                    broad_input_dims[i] : broad_mask_dims[i]
    }

    ctx->SetOutputDim("Out", framework::make_ddim(output_dims))
    */
  }
};

class MaskedSelectOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("input",
             "(Tensor, default Tensor<float>), "
             "the input tensor of MaskedSelectOp.");

    AddInput("mask",
             "(Tensor, default Tensor<bool>), "
             "the tensor containing the binary mask to index.");

    AddOutput("Out",
              "(Tensor, default Tensor<float>), the output of "
              "MaskedSelectOp.");

    AddComment(R"DOC(
		Masked select operator
    This operator obtains a new 1-D tensor whose elements are selected acrroding to the boolean mask which is a bool tensor. The shapes of the mask and the input don't need to match , but they must be broadcastable.)DOC");
  }
};

class MaskedSelectGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType("masked_select_grad");
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("input"), InputGrad("input"));
    op->SetOutput(framework::GradVarName("mask"), InputGrad("mask"));

    return std::unique_ptr<framework::OpDesc>(op);
  }
};

class MaskedSelectGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@Grad) should not be null");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("input")),
                   "Output(input@Grad) should not be null");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("mask")),
                   "Output(mask@Grad) should not be null");

    auto do_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    // check dim??

    ctx->SetOutputDim(framework::GradVarName("input"), do_dims);
    ctx->SetOutputDim(framework::GradVarName("mask"), do_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(masked_select, ops::MaskedSelectOp, ops::MaskedSelectOpMaker,
                  ops::MaskedSelectGradMaker);

REGISTER_OPERATOR(masked_select_grad, ops::MaskedSelectGradOp);

REGISTER_OP_CPU_KERNEL(
    masked_select,
    ops::MaskedSelectOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MaskedSelectOpKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    masked_select_grad,
    ops::MaskedSelectGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MaskedSelectGradOpKernel<paddle::platform::CPUDeviceContext, double>);
