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

#include <fstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/dynload/nvjpeg.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

template <typename T>
class CPUDecodeJpegKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // TODO(LieLinJiang): add cpu implement.
    PADDLE_THROW(platform::errors::Unimplemented(
        "DecodeJpeg op only supports GPU now."));
  }
};

class DecodeJpegOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "DecodeJpeg");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "DecodeJpeg");

    auto out_dims = std::vector<int>(1, -1);
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
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

class DecodeJpegOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Diagonal values of square matrix. It is a tensor with rank 1.");
    AddOutput("Out", "The output tensor of DecodeJpeg op");
    AddComment(R"DOC(
This operator initializes a tensor with random integers sampled from a
uniform distribution. The random result is in set [low, high).
)DOC");
    AddAttr<std::string>(
        "mode",
        "(string, default \"unchanged\"), The read mode used "
        "for optionally converting the image, can be \"unchanged\" "
        ",\"gray\" , \"rgb\" .")
        .SetDefault("unchanged");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    decode_jpeg, ops::DecodeJpegOp, ops::DecodeJpegOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(decode_jpeg, ops::CPUDecodeJpegKernel<uint8_t>)
