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
    OP_INOUT_CHECK(ctx->HasInput("attn_mask"), "Input", "attn_mask",
                   "MHADataPrepOp");

    auto attn_mask_dims = ctx->GetInputDim("attn_mask");

    size_t qo_kv_size = attn_mask_dims[0] * 2;
    size_t low_high_win_size = 0;
    if (attn_mask_dims.size() == 2) {
      low_high_win_size = attn_mask_dims[1] * 2;
    } else if (attn_mask_dims.size() == 4) {
      low_high_win_size = attn_mask_dims[3] * 2;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The dimension of attn mask to mha_data_prepare op "
          "should be equal to 2 or 4 . But received "
          "dimensions = %d.",
          attn_mask_dims.size()));
    }

    std::vector<int64_t> qkvo_output_dims(1, qo_kv_size);
    ctx->SetOutputDim("qo_kv_seqlen", phi::make_ddim(qkvo_output_dims));
    ctx->SetOutputDim("qo_kv_seqlen_host", phi::make_ddim(qkvo_output_dims));

    std::vector<int64_t> lo_hi_output_dims(1, low_high_win_size);
    ctx->SetOutputDim("low_high_windows_host",
                      phi::make_ddim(lo_hi_output_dims));
  }
};

class MHADataPrepOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("attn_mask", "[batch, heads, seqlen, seqlen]");

    AddOutput("qo_kv_seqlen", "");
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
