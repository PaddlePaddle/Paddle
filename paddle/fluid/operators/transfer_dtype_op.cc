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

#include "paddle/fluid/operators/transfer_dtype_op.h"

#include <string>

namespace paddle {
namespace framework {
class OpDesc;
class InferShapeContext;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class TransferDtypeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "TransferDtype");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "TransferDtype");

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    // dtype is not important
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return expected_kernel_type;
  }
};

class TransferDtypeKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *x = ctx.InputVar("X");
    auto *out = ctx.OutputVar("Out");
    auto &dev_ctx = ctx.device_context();
    auto dst_dtype = ctx.Attr<int>("dst_dtype");
    TransferDtypeFunctor(x, out, dev_ctx, dst_dtype)();
  }
};

class TransferDtypeOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) The input Tensor");
    AddOutput("Out", "(LoDTensor) The Output Tensor with desired layout");
    AddAttr<int>("dst_dtype",
                 "BOOL=1, INT8, UINT8, INT16, INT32, UINT32, INT64, UINT64, "
                 "BFLOAT16, FLOAT16, UINT16,"
                 "FLOAT32, FLOAT64, COMPLEX64, COMPLEX128");
    AddComment(R"DOC(
    TransferDtype Operator)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(
    transfer_dtype, ops::TransferDtypeOp, ops::TransferDtypeOpProtoMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

// dtype is not important
REGISTER_OP_CPU_KERNEL_FUNCTOR(transfer_dtype, float, ops::TransferDtypeKernel);
