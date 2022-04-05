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

#include "paddle/fluid/operators/data/file_label_loader_op.h"

namespace paddle {
namespace operators {
namespace data {

class FileLabelLoaderOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Indices"), true,
                      platform::errors::InvalidArgument(
                          "Input(Indices) of ReadFileLoaderOp is null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Label"), true,
                      platform::errors::InvalidArgument(
                          "Output(Label) of ReadFileLoaderOp is null."));

    auto dim_indices = ctx->GetInputDim("Indices");
    PADDLE_ENFORCE_EQ(dim_indices.size(), 1,
                      platform::errors::InvalidArgument(
                          "Input(Indices) should be a 1-D Tensor"));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    return framework::OpKernelType(framework::proto::VarType::UINT8,
                                   platform::CPUPlace());
  }
};

class FileLabelLoaderOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Indices", "The batch indices of input samples");
    AddOutput("Image", "The output image tensor of ReadFileLoader op")
        .AsDuplicable();
    AddOutput("Label", "The output label tensor of ReadFileLoader op");
    AddAttr<std::string>("data_root", "Path of root directory of dataset");
    AddComment(R"DOC(
      This operator read ImageNet format dataset for :attr:`data_root` with
      given indices.
      There are 2 outputs:
      1. Image: a list of Tensor which holds the image bytes data
      2. Label: a Tensor with shape [N] and dtype as int64, N is the batch
      size, which is the length of input indices.
)DOC");
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::data;

REGISTER_OPERATOR(
    file_label_loader, ops::FileLabelLoaderOp, ops::FileLabelLoaderOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(file_label_loader,
                       ops::FileLabelLoaderCPUKernel<uint8_t>)
