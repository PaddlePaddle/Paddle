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
      uint64_t tmp_var_key = 0;
      for (size_t i = 0; i < num_inputs; ++i) {
        const auto dims = ins_dims[i];
        int rank = dims.size();
        int cur_batch_size = 0;
        framework::Variable* x_var =
            PADDLE_GET(framework::Variable*, inputs_tensor[i]);
        const auto x_tensor = x_var->GetMutable<phi::DenseTensor>();
        tmp_var_key += (uint64_t)(x_tensor);
        const auto& x_lod = x_tensor->lod();
        if (x_lod.size() > 0) {
          cur_batch_size = x_lod[0].size() - 1;
        } else {
          cur_batch_size = x_tensor->dims()[0];
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

      // 准备lod的gpu数据，不然放到computer里面会拖垮性能
      {
        auto scope = ctx->GetScopePtr();
        auto& child_scope = scope->NewScope();
        std::string var_name = "FusedSeqpoolCVMOp_";
        var_name.append(std::to_string(tmp_var_key));
        auto var = child_scope.Var(var_name);
        paddle::framework::GpuPinnedVector* pin_ptr =
            var->GetMutable<paddle::framework::GpuPinnedVector>();

        std::vector<size_t> mix_lods;
        mix_lods.reserve(num_inputs * (batch_size + 1));
        for (size_t i = 0; i < num_inputs; ++i) {
          framework::Variable* x_var =
              PADDLE_GET(framework::Variable*, inputs_tensor[i]);
          const auto& x_tensor = x_var->Get<phi::DenseTensor>();
          const auto& x_lod = x_tensor.lod();
          if (x_lod.size() != 0) {
            PADDLE_ENFORCE_EQ(x_lod.size(),
                              1,
                              platform::errors::PreconditionNotMet(
                                  "The lod size of all input should be 1, "
                                  "please cheack"));
            PADDLE_ENFORCE_EQ(
                x_lod[0].size(),
                batch_size + 1,
                platform::errors::PreconditionNotMet(
                    "The lod[0] size of all input should be batch_size + 1, "
                    "please cheack"));
            mix_lods.insert(mix_lods.end(), x_lod[0].begin(), x_lod[0].end());
          } else {
            mix_lods.push_back(0);
            for (int i = 0; i < x_tensor.dims()[0]; i++) {
              mix_lods.push_back(i + 1);
            }
          }
        }
        pin_ptr->cpu_to_pinedcpu(mix_lods.data(),
                                 mix_lods.size() * sizeof(size_t));
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
    bool flag = 0;
    for (auto* input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = framework::TransToProtoVarType(input->dtype());
        flag = 1;
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

    // 准备lod的gpu数据，不然放到computer里面会拖垮性能
    if (ctx->IsRuntime()) {
      auto inputs_tensor = ctx->GetOutputVarPtrs(framework::GradVarName("X"));
      size_t num_inputs = inputs_tensor.size();
      uint64_t tmp_var_key = 0;
      framework::Variable* x_var =
          PADDLE_GET(framework::Variable*, inputs_tensor[0]);
      const phi::DenseTensor* x_tensor = x_var->GetMutable<phi::DenseTensor>();
      int batch_size = x_tensor->lod().size() ? x_tensor->lod()[0].size() - 1
                                              : x_tensor->dims()[0];

      std::vector<size_t> mix_lods;
      mix_lods.reserve(num_inputs * (batch_size + 1));
      for (size_t i = 0; i < num_inputs; i++) {
        x_var = PADDLE_GET(framework::Variable*, inputs_tensor[i]);
        x_tensor = x_var->GetMutable<phi::DenseTensor>();
        tmp_var_key += (uint64_t)(x_tensor);
        const auto& x_lod = x_tensor->lod();
        if (x_lod.size() != 0) {
          PADDLE_ENFORCE_EQ(x_lod.size(),
                            1,
                            platform::errors::PreconditionNotMet(
                                "The lod size of all in_grad should be 1, "
                                "please cheack"));
          PADDLE_ENFORCE_EQ(
              x_lod[0].size(),
              batch_size + 1,
              platform::errors::PreconditionNotMet(
                  "The lod[0] size of all in_grad should be batch_size + 1, "
                  "please cheack"));
          mix_lods.insert(mix_lods.end(), x_lod[0].begin(), x_lod[0].end());
        } else {
          mix_lods.push_back(0);
          for (int i = 0; i < x_tensor->dims()[0]; i++) {
            mix_lods.push_back(i + 1);
          }
        }
        int cur_batch_size = x_tensor->lod().size()
                                 ? x_tensor->lod()[0].size() - 1
                                 : x_tensor->dims()[0];
        PADDLE_ENFORCE_EQ(batch_size,
                          cur_batch_size,
                          platform::errors::PreconditionNotMet(
                              "The batch size of all in_grad should be same, "
                              "please cheack, last batchsize is %d, current "
                              "batchsize is %d",
                              batch_size,
                              cur_batch_size));
      }
      PADDLE_ENFORCE_EQ(mix_lods.size(),
                        num_inputs * (batch_size + 1),
                        platform::errors::PreconditionNotMet("please cheack"));

      std::string var_name = "FusedSeqpoolCVMGradOp_";
      var_name.append(std::to_string(tmp_var_key));
      auto scope = ctx->GetScopePtr();
      auto& child_scope = scope->NewScope();
      auto var = child_scope.Var(var_name);
      paddle::framework::GpuPinnedVector* pin_ptr =
          var->GetMutable<paddle::framework::GpuPinnedVector>();
      pin_ptr->cpu_to_pinedcpu(mix_lods.data(),
                               mix_lods.size() * sizeof(size_t));
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
namespace plat = paddle::platform;

REGISTER_OPERATOR(fused_seqpool_cvm,
                  ops::FusedSeqpoolCVMOp,
                  ops::FusedSeqpoolCVMOpMaker,
                  ops::FusedSeqpoolCVMGradOpMaker<paddle::framework::OpDesc>,
                  ops::FusedSeqpoolCVMGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_seqpool_cvm_grad, ops::FusedSeqpoolCVMGradOp)

REGISTER_OP_CPU_KERNEL(fused_seqpool_cvm,
                       ops::FusedSeqpoolCVMOpCPUKernel<float>)
REGISTER_OP_CPU_KERNEL(fused_seqpool_cvm_grad,
                       ops::FusedSeqpoolCVMGradOpCPUKernel<float>)
