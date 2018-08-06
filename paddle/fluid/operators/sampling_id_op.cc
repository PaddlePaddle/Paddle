/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/sampling_id_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class SamplingIdOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of RowConvOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of RowConvOp should not be null.");

    auto input_dims = ctx->GetInputDim("X");

    framework::DDim dims = input_dims;
    ctx->SetOutputDim("Out", dims);
    ctx->ShareLoD("X", "Out");
  }
};

class SamplingIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of softmax. "
             "2-D with shape [batch_size, input_feature_dimensions].");
    AddOutput("Out", "Sliced data tensor.");

    AddComment(R"DOC(
SamplingId Operator.
  @brief A layer for sampling id from multinomial distribution from the
 input layer. Sampling one id for one sample. The result is stored in
 output_.ids.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sampling_id,
    ops::SamplingIdKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SamplingIdKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SamplingIdKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SamplingIdKernel<paddle::platform::CUDADeviceContext, int64_t>);
