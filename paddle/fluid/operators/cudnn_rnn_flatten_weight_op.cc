// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/cudnn_rnn_flatten_weight_op.h"

namespace paddle {
namespace operators {

class CudnnRnnFlattenWeightOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Output", "Weight",
                   "Cudnn_rnn_flatten_weight");
    OP_INOUT_CHECK(ctx->HasOutput("WeightHh"), "Input", "WeightHh",
                   "Cudnn_rnn_flatten_weight");
    OP_INOUT_CHECK(ctx->HasOutput("WeightIh"), "Input", "WeightIh",
                   "Cudnn_rnn_flatten_weight");
    OP_INOUT_CHECK(ctx->HasOutput("BiasHh"), "Input", "BiasHh",
                   "Cudnn_rnn_flatten_weight");
    OP_INOUT_CHECK(ctx->HasOutput("BiasIh"), "Input", "BiasIh",
                   "Cudnn_rnn_flatten_weight");
  }
};

class CudnnRnnFlattenWeightOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Weight",
             "(Tensor), share data with WeightHh, WeightIh, BiasHh, BiasIh.");
    AddOutput("WeightHh",
              "(Tensor List), stores weight_hh and share data with W. ")
        .AsDuplicable();
    AddOutput("WeightIh",
              "(Tensor List), stores weight_ih and share data with W.")
        .AsDuplicable();
    AddOutput("BiasHh", "(Tensor List), stores bias_hh and share data with W.")
        .AsDuplicable();
    AddOutput("BiasIh", "(Tensor List), stores bias_ih and share data with W. ")
        .AsDuplicable();
    AddAttr<bool>("is_bidirec",
                  "If it is bidirectional rnn, this will affect the shape of "
                  "the Out, LastH, and LastC.")
        .SetDefault(false);
    AddAttr<int>("hidden_size", "hidden size of the LSTM").SetDefault(100);
    AddAttr<int>("input_size", "input size of the LSTM").SetDefault(100);
    AddAttr<int>("num_layers", "the total layer number of the LSTM")
        .SetDefault(1);
    AddComment(R"DOC(
      Cudnn_rnn_flatten_weight Operator.
      Designed to share input and output memory.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(cudnn_rnn_flatten_weight, ops::CudnnRnnFlattenWeightOp,
                  ops::CudnnRnnFlattenWeightOpMaker);

REGISTER_OP_CPU_KERNEL(
    cudnn_rnn_flatten_weight,
    ops::CudnnRnnFlattenWeightKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CudnnRnnFlattenWeightKernel<paddle::platform::CPUDeviceContext,
                                     double>,
    ops::CudnnRnnFlattenWeightKernel<paddle::platform::CPUDeviceContext, int>,
    ops::CudnnRnnFlattenWeightKernel<paddle::platform::CPUDeviceContext,
                                     int64_t>);
