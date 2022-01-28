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

#include "paddle/fluid/operators/data/batch_decode_op.h"

namespace paddle {
namespace operators {
namespace data {

class BatchDecodeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "DecodeJpeg");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "DecodeJpeg");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::UINT8, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (var_name == "X") {
      return expected_kernel_type;
    }

    return framework::OpKernelType(tensor.type(), tensor.place(),
                                   tensor.layout());
  }
};

class BatchDecodeInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    ctx->SetOutputType("Out", framework::proto::VarType::LOD_TENSOR_ARRAY,
                       framework::ALL_ELEMENTS);
  }
};

class BatchDecodeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "A one dimensional uint8 tensor containing the raw bytes "
             "of the JPEG image. It is a tensor with rank 1.");
    AddOutput("Out", "The output tensor of DecodeJpeg op");
    AddComment(R"DOC(
This operator decodes a JPEG image into a 3 dimensional RGB Tensor 
or 1 dimensional Gray Tensor. Optionally converts the image to the 
desired format. The values of the output tensor are uint8 between 0 
and 255.
)DOC");
    AddAttr<int>("num_threads", "Path of the file to be readed.")
      .SetDefault(2);
    AddAttr<std::string>(
        "mode",
        "(string, default \"unchanged\"), The read mode used "
        "for optionally converting the image, can be \"unchanged\" "
        ",\"gray\" , \"rgb\" .")
        .SetDefault("unchanged");
    AddAttr<int>("local_rank",
                 "(int)"
                 "The index of the op to start execution");
    AddAttr<int64_t>("program_id",
                     "(int64_t)"
                     "The unique hash id used as cache key for "
                     "decode thread pool");
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    batch_decode, ops::data::BatchDecodeOp, ops::data::BatchDecodeOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(batch_decode, ops::data::CPUBatchDecodeKernel<uint8_t>)
