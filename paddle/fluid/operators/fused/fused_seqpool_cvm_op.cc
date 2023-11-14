/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_seqpool_cvm_op.h"

#include <string>
namespace paddle {
namespace operators {

class FusedSeqpoolCVMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(
        ctx->Inputs("X").size(),
        1UL,
        platform::errors::InvalidArgument(
            "Inputs(X) of FusedSeqpoolCVMOp should not be empty."));
    PADDLE_ENFORCE_GE(
        ctx->Outputs("Out").size(),
        1UL,
        platform::errors::InvalidArgument(
            "Outputs(Out) of FusedSeqpoolCVMOp should not be empty."));

    auto cvm_dims = ctx->GetInputDim("CVM");
    PADDLE_ENFORCE_EQ(
        cvm_dims.size(),
        2UL,
        platform::errors::InvalidArgument("Input(CVM)'s rank should be 2."));
    PADDLE_ENFORCE_EQ(
        cvm_dims[1],
        2UL,
        platform::errors::InvalidArgument("The 2nd dimension of "
                                          "Input(CVM) should be 2."));

    auto ins_dims = ctx->GetInputsDim("X");
    const int cvm_offset = ctx->Attrs().Get<int>("cvm_offset");
    const size_t num_inputs = ins_dims.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.resize(num_inputs);
    bool use_cvm = ctx->Attrs().Get<bool>("use_cvm");

    PADDLE_ENFORCE_GT(num_inputs,
                      0UL,
                      platform::errors::InvalidArgument(
                          "Input tensors count should be greater than 0, "
                          "but received value is %d.",
                          num_inputs));

    // The output height should be confirmed in Compute,
    // since input lod is not accessible here.
    PADDLE_ENFORCE_EQ(ins_dims[0].size(),
                      2,
                      platform::errors::InvalidArgument(
                          "The dims size of first input should be equal to 2, "
                          "but received value is %d.",
                          ins_dims[0].size()));

    if (ctx->IsRuntime()) {
      int batch_size = -1;
      auto inputs_tensor = ctx->GetInputVarPtrs("X");
      for (size_t i = 0; i < num_inputs; ++i) {
        const auto dims = ins_dims[i];
        int rank = dims.size();
        int cur_batch_size = 0;
        framework::Variable* x_var =
            PADDLE_GET(framework::Variable*, inputs_tensor[i]);
        const auto& x_tensor = x_var->Get<phi::DenseTensor>();
        const auto& x_lod = x_tensor.lod();
        if (!x_lod.empty()) {
          cur_batch_size = static_cast<int>(x_lod[0].size() - 1);
        } else {
          cur_batch_size = static_cast<int>(x_tensor.dims()[0]);
        }
        if (batch_size == -1) {
          batch_size = cur_batch_size;
        } else {
          PADDLE_ENFORCE_EQ(batch_size,
                            cur_batch_size,
                            platform::errors::PreconditionNotMet(
                                "The batch size of all input should be same, "
                                "please check, last batch_size is %d, current "
                                "batch_size is %d",
                                batch_size,
                                cur_batch_size));
        }
        std::vector<int64_t> out_dim;
        if (use_cvm) {
          out_dim = {batch_size, dims[rank - 1]};
        } else {
          out_dim = {batch_size, dims[rank - 1] - cvm_offset};
        }
        outs_dims[i] = phi::make_ddim(out_dim);
      }
    } else {
      for (size_t i = 0; i < num_inputs; ++i) {
        const auto dims = ins_dims[i];
        int rank = dims.size();
        if (use_cvm) {
          PADDLE_ENFORCE_GT(
              dims[rank - 1],
              2,
              platform::errors::InvalidArgument(
                  "Shape error in %lu id, the last dimension(embedding) of the "
                  "'X' tensor must be larger than 2.",
                  i));
        }
        // input lod is not accessible here
        std::vector<int64_t> out_dim;
        if (use_cvm) {
          out_dim = {-1, dims[rank - 1]};
        } else {
          out_dim = {-1, dims[rank - 1] - cvm_offset};
        }
        outs_dims[i] = phi::make_ddim(out_dim);
      }
    }
    ctx->SetOutputsDim("Out", outs_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<phi::DenseTensor>("X");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = false;
    for (auto* input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = framework::TransToProtoVarType(input->dtype());
        flag = true;
        break;
      }
    }
    PADDLE_ENFORCE_EQ(flag,
                      1,
                      platform::errors::InvalidArgument(
                          "All Inputs of fused_seqpool_cvm OP are Empty!"));
    return phi::KernelKey(input_data_type, ctx.GetPlace());
    // return phi::KernelKey(framework::proto::VarType::FP32,
    //                                ctx.device_context());
    // return phi::KernelKey(
    //   OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FusedSeqpoolCVMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(vector<phi::DenseTensor>) The input tensors of"
             " operator.")
        .AsDuplicable();
    AddInput("CVM",
             "(Tensor),  a 2-D Tensor with shape [N x 2], where N is the batch "
             "size, 2 is show and click.");
    AddOutput("Out",
              "(vector<Tensor>) The output of Op does not contain LoD "
              "information.")
        .AsDuplicable();
    AddAttr<std::string>("pooltype",
                         "(string, default 'SUM') the pooling pooltype of "
                         "SequencePoolOp, only support SUM now.")
        .SetDefault("SUM")
        .InEnum({"SUM"});
    AddAttr<float>("pad_value",
                   "(float, default 0.0) The value to pad for empty sequence.")
        .SetDefault(0.0);
    AddAttr<bool>("use_cvm", "bool, use cvm or not").SetDefault(true);
    AddAttr<int>("cvm_offset", "(int, default 2)").SetDefault(2);

    AddComment(R"DOC(
Fuse multiple pairs of Sequence Pool and CVM Operator.

)DOC");
  }
};

class FusedSeqpoolCVMGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto og_dims = ctx->GetInputsDim(framework::GradVarName("Out"));
    auto x_dims = ctx->GetInputsDim("X");
    auto cvm_dims = ctx->GetInputDim("CVM");
    const int cvm_offset = ctx->Attrs().Get<int>("cvm_offset");
    bool use_cvm = ctx->Attrs().Get<bool>("use_cvm");

    PADDLE_ENFORCE_EQ(
        cvm_dims.size(),
        2,
        platform::errors::InvalidArgument("Input(CVM)'s rank should be 2."));

    for (size_t i = 0; i < og_dims.size(); i++) {
      PADDLE_ENFORCE_EQ(
          og_dims[i].size(),
          x_dims[i].size(),
          platform::errors::InvalidArgument(
              "The rank of output grad must equal to Input(X). But "
              "received: input rank %u, input shape [%s].",
              og_dims[i].size(),
              og_dims[i]));
      if (use_cvm) {
        auto o_dim = og_dims[i][og_dims[i].size() - 1];
        PADDLE_ENFORCE_EQ(
            o_dim,
            x_dims[i][og_dims[i].size() - 1],
            platform::errors::InvalidArgument(
                "The dimension mismatch between Input(OUT@GRAD) and "
                "Input(X). Received Input(OUT@GRAD): input rank %u, "
                "input shape [%s]; received Input(X): input rank %u, "
                "input shape [%s].",
                og_dims[i].size(),
                og_dims[i],
                x_dims[i].size(),
                x_dims[i]));
      } else {
        PADDLE_ENFORCE_EQ(
            og_dims[i][og_dims[i].size() - 1],
            x_dims[i][og_dims[i].size() - 1] - cvm_offset,
            platform::errors::InvalidArgument(
                "The dimension mismatch between Input(OUT@GRAD) and "
                "Input(X). Received Input(OUT@GRAD): input rank %u, "
                "input shape [%s]; received Input(X): input rank %u, "
                "input shape [%s].",
                og_dims[i].size(),
                og_dims[i],
                x_dims[i].size(),
                x_dims[i]));
      }
    }
    for (size_t i = 0; i < x_dims.size(); ++i) {
      ctx->ShareLoD("X", framework::GradVarName("X"), i, i);
      ctx->ShareDim("X", framework::GradVarName("X"), i, i);
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(
                              ctx, framework::GradVarName("Out")),
                          ctx.GetPlace());
  }
};

template <typename T>
class FusedSeqpoolCVMGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("fused_seqpool_cvm_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    op_desc_ptr->SetInput("CVM", this->Input("CVM"));

    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"),
                           this->InputGrad("X", false));
    op_desc_ptr->SetOutput(framework::GradVarName("CVM"),
                           this->InputGrad("CVM"));
    op_desc_ptr->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(fused_seqpool_cvm,
                  ops::FusedSeqpoolCVMOp,
                  ops::FusedSeqpoolCVMOpMaker,
                  ops::FusedSeqpoolCVMGradOpMaker<paddle::framework::OpDesc>,
                  ops::FusedSeqpoolCVMGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_seqpool_cvm_grad, ops::FusedSeqpoolCVMGradOp)

PD_REGISTER_STRUCT_KERNEL(fused_seqpool_cvm,
                          CPU,
                          ALL_LAYOUT,
                          ops::FusedSeqpoolCVMOpCPUKernel,
                          float) {}
PD_REGISTER_STRUCT_KERNEL(fused_seqpool_cvm_grad,
                          CPU,
                          ALL_LAYOUT,
                          ops::FusedSeqpoolCVMGradOpCPUKernel,
                          float) {}
