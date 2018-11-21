// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/sequence_ops/sequence_concat_op.h"
#include <vector>

namespace paddle {
namespace operators {

class SeqConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The inputs of sequence concat op").AsDuplicable();
    AddOutput("Out", "The output of sequence concat op");
    AddComment(
        "Sequence Concat Op\n"
        "It will concat LoD tensors by its sequence information.\n"
        "For example:\n"
        "  LoD of X1 = [0, 3, 7]\n"
        "  LoD of X2 = [0, 7, 9]\n"
        "  Result LoD is [0, (3+7), (7+9)]\n"
        "            i.e.[0, 10, 16]\n");
  }
};

class SeqConcatShapeInferer : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInputs("X"),
                   "Input(X) of Sequence Concat Op should not be null.");
    PADDLE_ENFORCE(context->HasOutput("Out"),
                   "Output(Out) of Sequence Concat Op should not be null.");

    PADDLE_ENFORCE_GT(context->Inputs("X").size(), 1,
                      "The number of input sequences is at least two.");
    auto x_dims = context->GetInputsDim("X");
    int64_t batch_size = 0;
    int64_t feature_size = 0;
    std::vector<int64_t> out_dims;
    for (auto &x_dim : x_dims) {
      if (out_dims.empty()) {
        out_dims = framework::vectorize(x_dim);
      }
      batch_size += x_dim[0];
      if (feature_size == 0) {
        feature_size = framework::product(x_dim) / x_dim[0];
      } else {
        PADDLE_ENFORCE_EQ(
            feature_size, framework::product(x_dim) / x_dim[0],
            "Inputs of sequence concat must have same feature size");
      }
    }
    if (batch_size < 0) {
      batch_size = -1;  // Normalize batch size for compile time.
    }
    out_dims[0] = batch_size;
    context->SetOutputDim("Out", framework::make_ddim(out_dims));
    if (!context->IsRuntime()) {  // Runtime LoD infershape will be computed
      // in Kernel.
      context->ShareLoD("X", "Out");
    }
  }
};

class SeqConcatGradShapeInferer : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    context->SetOutputsDim(framework::GradVarName("X"),
                           context->GetInputsDim("X"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace op = paddle::operators;

REGISTER_OPERATOR(sequence_concat, paddle::framework::OperatorWithKernel,
                  op::SeqConcatOpMaker, op::SeqConcatShapeInferer,
                  paddle::framework::DefaultGradOpDescMaker<false>);
template <typename T>
using Kernel = op::SeqConcatKernel<paddle::platform::CPUDeviceContext, T>;
REGISTER_OP_CPU_KERNEL(sequence_concat, Kernel<float>, Kernel<double>,
                       Kernel<int64_t>);

REGISTER_OPERATOR(sequence_concat_grad, paddle::framework::OperatorWithKernel,
                  op::SeqConcatGradShapeInferer);
template <typename T>
using GradKernel =
    op::SeqConcatGradKernel<paddle::platform::CPUDeviceContext, T>;
REGISTER_OP_CPU_KERNEL(sequence_concat_grad, GradKernel<float>,
                       GradKernel<double>, GradKernel<int64_t>);
