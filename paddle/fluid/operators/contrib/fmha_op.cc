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

/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "paddle/fluid/operators/contrib/fmha_op.h"
#include <memory>
#include <string>

#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class FMHAOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FMHA");
    OP_INOUT_CHECK(ctx->HasInput("Seqlen"), "Input", "Seqlen", "FMHA");
    OP_INOUT_CHECK(ctx->HasInput("Cu_seqlen"), "Input", "Cu_seqlen", "FMHA");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "FMHA");
    OP_INOUT_CHECK(ctx->HasOutput("SoftmaxMask"), "Output", "SoftmaxMask",
                   "FMHA");

    const auto x_dims = ctx->GetInputDim("X");
    const auto x_size = x_dims.size();

    PADDLE_ENFORCE_EQ(x_dims[THREE_DIMS], 3,
                      platform::errors::InvalidArgument(
                          "The input dims[%d] should be equal to 3."
                          "But receive %d.",
                          THREE_DIMS, x_dims[THREE_DIMS]));

    PADDLE_ENFORCE_EQ(x_size, 3,
                      platform::errors::InvalidArgument(
                          "The input dims size should be equal to 3 with "
                          "shape [SLEN, 3, HIDDEN]. But receive input dims "
                          "size %d.",
                          x_size));

    const auto max_slen = x_dims[SLEN_DIMS];
    const auto hidden = x_dims[HIDDEN_DIMS];

    // For demo, use num_head = 16
    constexpr int NUM_HEAD = 16;

    framework::DDim out_dims = framework::make_ddim({max_slen, hidden});
    framework::DDim softmax_mask_dims =
        framework::make_ddim({NUM_HEAD, max_slen, max_slen});

    ctx->SetOutputDim("Out", out_dims);
    ctx->SetOutputDim("SoftmaxMask", softmax_mask_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

class FMHAOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The packed QKV tensor, tensor shape is "
             "strictly limited to [SLEN, 3, HIDDEN].");
    AddInput("Seqlen",
             "(Tensor) Sequence length list in a batch, "
             "shape should be equal to [batch]");
    AddInput("Cu_seqlen",
             "(Tensor) Cumulative sequence length in a batch, "
             "shape should be equal to [batch + 1]");
    AddOutput("Out",
              "(Tensor) The output tensor with shape "
              "[SLEN, HIDDEN].");
    AddOutput("SoftmaxMask",
              "(Tensor) Softmax output for backward. Shape is "
              "[NUM_HEAD, SLEN, SLEN].");
    AddComment(R"DOC(FMHA Operator)DOC");
  }
};

class FMHAOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FMHAGrad");
    OP_INOUT_CHECK(ctx->HasInput("SoftmaxMask"), "Input", "SoftmaxMask",
                   "FMHAGrad");
    OP_INOUT_CHECK(ctx->HasInput("Seqlen"), "Input", "Seqlen", "FMHAGrad");
    OP_INOUT_CHECK(ctx->HasInput("Cu_seqlen"), "Input", "Cu_seqlen",
                   "FMHAGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "FMHAGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"), "FMHAGrad");
    auto out_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

template <typename T>
class FMHAGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("fmha_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Seqlen", this->Input("Seqlen"));
    grad_op->SetInput("Cu_seqlen", this->Input("Cu_seqlen"));
    grad_op->SetInput("SoftmaxMask", this->Output("SoftmaxMask"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fmha, ops::FMHAOp, ops::FMHAOpMaker,
                  ops::FMHAGradOpMaker<paddle::framework::OpDesc>,
                  ops::FMHAGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fmha_grad, ops::FMHAOpGrad);
REGISTER_OP_CPU_KERNEL(
    fmha, ops::FMHAKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    fmha_grad, ops::FMHAGradKernel<paddle::platform::CPUDeviceContext, float>);
