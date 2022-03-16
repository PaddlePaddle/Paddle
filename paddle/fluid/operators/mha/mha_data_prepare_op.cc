/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mha/mha_data_prepare_op.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class MHADataPrepOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("qo_kv_seqlen"), "Input", "qo_kv_seqlen",
                   "MHADataPrepOp");
    OP_INOUT_CHECK(ctx->HasInput("low_high_windows"), "Input",
                   "low_high_windows", "MHADataPrepOp");

    auto qkvo_input_dims = ctx->GetInputDim("qo_kv_seqlen");
    std::vector<int64_t> qkvo_output_dims;
    for (int i = 0; i < qkvo_input_dims.size(); ++i) {
      qkvo_output_dims.push_back(qkvo_input_dims[i]);
    }
    ctx->SetOutputDim("qo_kv_seqlen_host", phi::make_ddim(qkvo_output_dims));

    auto lo_hi_input_dims = ctx->GetInputDim("low_high_windows");
    std::vector<int64_t> lo_hi_output_dims;
    for (int i = 0; i < lo_hi_input_dims.size(); ++i) {
      lo_hi_output_dims.push_back(lo_hi_input_dims[i]);
    }
    ctx->SetOutputDim("low_high_windows_host",
                      phi::make_ddim(lo_hi_output_dims));
  }
};

class MHADataPrepOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("qo_kv_seqlen", "");
    AddInput("low_high_windows", "");

    AddOutput("qo_kv_seqlen_host", "");
    AddOutput("low_high_windows_host", "");

    AddComment(R"DOC(MHA Sequence data preparation OP)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mha_data_prepare, ops::MHADataPrepOp,
                  ops::MHADataPrepOpMaker);

namespace plat = paddle::platform;
