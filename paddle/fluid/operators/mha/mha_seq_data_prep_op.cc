/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2021 NVIDIA Corporation. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mha/mha_seq_data_prep_op.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class MHASeqDataPrepOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("QKVO_seqlen"), "Input", "QKVO_seqlen",
                   "MHASeqDataPrep");
    OP_INOUT_CHECK(ctx->HasInput("lo_hi_windows"), "Input", "lo_hi_windows",
                   "MHASeqDataPrep");

    auto qkvo_input_dims = ctx->GetInputDim("QKVO_seqlen");
    std::vector<int64_t> qkvo_output_dims;
    for (int i = 0; i < qkvo_input_dims.size(); ++i) {
      qkvo_output_dims.push_back(qkvo_input_dims[i]);
    }
    ctx->SetOutputDim("QKVO_seqlen_host",
                      framework::make_ddim(qkvo_output_dims));

    auto lo_hi_input_dims = ctx->GetInputDim("lo_hi_windows");
    std::vector<int64_t> lo_hi_output_dims;
    for (int i = 0; i < lo_hi_input_dims.size(); ++i) {
      lo_hi_output_dims.push_back(lo_hi_input_dims[i]);
    }
    ctx->SetOutputDim("lo_hi_windows_host",
                      framework::make_ddim(lo_hi_output_dims));
  }
};

class MHASeqDataPrepOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("QKVO_seqlen", "(Tensor), QKVO_seqlen");
    AddInput("lo_hi_windows", "(Tensor), lo_hi_windows");

    AddOutput("QKVO_seqlen_host", "(Tensor), QKVO_seqlen_host");
    AddOutput("lo_hi_windows_host", "(Tensor), lo_hi_windows_host");

    AddComment(R"DOC(MHA Sequence data preparation OP Test)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mha_seq_data_prep, ops::MHASeqDataPrepOp,
                  ops::MHASeqDataPrepOpMaker);

namespace plat = paddle::platform;

REGISTER_OP_CPU_KERNEL(
    mha_seq_data_prep,
    ops::MHASeqDataPrepKernel<paddle::platform::CPUDeviceContext, int32_t>);
