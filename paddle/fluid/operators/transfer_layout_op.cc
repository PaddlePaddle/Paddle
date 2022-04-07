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

#include "paddle/fluid/operators/transfer_layout_op.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

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

class TransferLayoutOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "TransferLayout");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "TransferLayout");

    auto dst_layout = ctx->Attrs().Get<int>("dst_layout");
    auto low_bound = static_cast<int>(framework::DataLayout::kAnyLayout);
    auto upper_bound = static_cast<int>(framework::DataLayout::kMKLDNN);
    PADDLE_ENFORCE_GE(
        dst_layout, low_bound,
        platform::errors::PreconditionNotMet(
            "Required dst_layout >= %d, but received dst_layout = %d",
            low_bound, dst_layout));
    PADDLE_ENFORCE_LE(
        dst_layout, upper_bound,
        platform::errors::PreconditionNotMet(
            "Required dst_layout <= %d, but received dst_layout = %d",
            upper_bound, dst_layout));

    // TODO(Aurelius84): Out's ddim is different with X because they have
    // different layout
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    // kernel's device type is decided by input tensor place
    auto *in = ctx.InputVar("X");
    auto *in_tensor = framework::GetLoDTensorOrSelectedRowsValueFromVar(*in);
    // NOTE(zhiqiu): hot fix, allow empty tensor of kMKLDNN layout to run this
    // op
    if (in_tensor->layout() != DataLayout::kMKLDNN) {
      PADDLE_ENFORCE_EQ(in_tensor->IsInitialized(), true,
                        platform::errors::PreconditionNotMet(
                            "The tensor of Input(X) is not initialized."));
    }
    auto place =
        in_tensor->IsInitialized() ? in_tensor->place() : platform::CPUPlace();

    // dtype is not important
    return framework::OpKernelType(framework::proto::VarType::FP32, place);
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   expected_kernel_type.data_layout_);
  }
};

class TransferLayoutInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};

class TransferLayoutKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *x = ctx.InputVar("X");
    auto *out = ctx.OutputVar("Out");
    auto &dev_ctx = ctx.device_context();
    auto src_layout = ctx.Attr<int>("src_layout");
    auto dst_layout = ctx.Attr<int>("dst_layout");
    auto input_name = ctx.InputName("X");
    TransferLayoutFunctor(x, out, dev_ctx, src_layout, dst_layout,
                          input_name)();
  }
};

class TransferLayoutOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) The input Tensor");
    AddOutput("Out", "(LoDTensor) The Output Tensor with desired layout");
    // NOTE(zhiqiu): in most case, the src_layout is not needed, the op can use
    // the layout
    // of input X. However, in some mkldnn kernel, the src layout computed by
    // GetKernelTypeForVar is different with the layout of tensor X.
    AddAttr<int>("src_layout",
                 "kAnyLayout = 0, kNHWC = 1, kNCHW = 2, kMKLDNN = 3, default "
                 "-1 means unspecified and use the tensor's layout.")
        .SetDefault(-1);
    AddAttr<int>("dst_layout",
                 "kAnyLayout = 0, kNHWC = 1, kNCHW = 2, kMKLDNN = 3");
    AddComment(R"DOC(
    TransferLayout Operator)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(
    transfer_layout, ops::TransferLayoutOp, ops::TransferLayoutOpProtoMaker,
    ops::TransferLayoutInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

// dtype is not important
REGISTER_OP_CPU_KERNEL_FUNCTOR(transfer_layout, float,
                               ops::TransferLayoutKernel);
REGISTER_OP_VERSION(transfer_layout)
    .AddCheckpoint(
        R"ROC(refine transfer_layout, add src_layout attribute)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "src_layout", "(int, the layout of the input tensor", -1));
