/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/crf_decoding_kernel.h"

namespace phi {

class CRFDecodingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Emission"), "Input", "Emission", "CRFDecoding");
    OP_INOUT_CHECK(
        ctx->HasInput("Transition"), "Input", "Transition", "CRFDecoding");
    OP_INOUT_CHECK(
        ctx->HasOutput("ViterbiPath"), "Output", "ViterbiPath", "CRFDecoding");

    auto emission_dims = ctx->GetInputDim("Emission");
    bool has_length = ctx->HasInput("Length");

    if (has_length) {
      PADDLE_ENFORCE_EQ(emission_dims.size(),
                        3,
                        phi::errors::InvalidArgument(
                            "The Input(Emission) should be a 3-D tensor. But "
                            "received: input rank %u, input shape [%s]. ",
                            emission_dims.size(),
                            emission_dims));
    } else {
      PADDLE_ENFORCE_EQ(emission_dims.size(),
                        2,
                        phi::errors::InvalidArgument(
                            "The Input(Emission) should be a 2-D tensor. But "
                            "received: input rank %u, input shape [%s].",
                            emission_dims.size(),
                            emission_dims));
    }

    auto transition_dims = ctx->GetInputDim("Transition");
    PADDLE_ENFORCE_EQ(transition_dims.size(),
                      2UL,
                      phi::errors::InvalidArgument(
                          "The Input(Transition) should be a 2-D tensor. But "
                          "received: input rank %u, input shape [%s].",
                          transition_dims.size(),
                          transition_dims));
    PADDLE_ENFORCE_EQ(
        transition_dims[0] - 2,
        transition_dims[1],
        phi::errors::InvalidArgument(
            "An invalid dimension for the Input(Transition), which should "
            "be a 2-D tensor with shape [(D + 2) x D]. But received: input "
            "rank %u, "
            "input shape [%s].",
            transition_dims.size(),
            transition_dims));
    if (ctx->IsRuntime() || (emission_dims[emission_dims.size() - 1] > 0 &&
                             transition_dims[transition_dims.size() - 1] > 0)) {
      PADDLE_ENFORCE_EQ(emission_dims[emission_dims.size() - 1],
                        transition_dims[transition_dims.size() - 1],
                        phi::errors::InvalidArgument(
                            "The last dimension of the Input(Emission) and the "
                            "Input(Transition) "
                            "should be equal to the tag number. But received "
                            "Input(Emission): rank "
                            "%u, shape [%s]; received Input(Transition): rank "
                            "%u, shape [%s].",
                            emission_dims.size(),
                            emission_dims,
                            transition_dims.size(),
                            transition_dims));
    }
    if (ctx->HasInput("Label")) {
      auto label_dims = ctx->GetInputDim("Label");
      if (ctx->HasInput("Length")) {
        PADDLE_ENFORCE_EQ(
            (label_dims.size() == 3UL && label_dims[2] == 1) ||
                label_dims.size() == 2UL,
            true,
            phi::errors::InvalidArgument(
                "The Input(Label) should be a 3-D tensor with last dimension "
                "fixed to 1 or a 2-D tensor in padding mode. But received: "
                "input "
                "rank %u, input shape [%s].",
                label_dims.size(),
                label_dims));
      } else {
        PADDLE_ENFORCE_EQ(
            (label_dims.size() == 2UL && label_dims[1] == 1) ||
                label_dims.size() == 1UL,
            true,
            phi::errors::InvalidArgument(
                "The Input(Label) should be a 2-D tensor with last "
                "dimension fixed to 1 or a 1-D tensor. But received: "
                "input rank %u, input shape [%s].",
                label_dims.size(),
                label_dims));
      }
      if (ctx->IsRuntime() || (emission_dims[0] > 0 && label_dims[0] > 0)) {
        PADDLE_ENFORCE_EQ(
            emission_dims[0],
            label_dims[0],
            phi::errors::InvalidArgument(
                "The first dimension of Input(Emission) and Input(Label) "
                "should be the same. But received Input(Emission): rank %u, "
                "shape [%s]; received Input(Label): rank %u, shape [%s].",
                emission_dims.size(),
                emission_dims,
                label_dims.size(),
                label_dims));
      }
    }

    ctx->ShareLoD("Emission", /*->*/ "ViterbiPath");
    if (has_length) {
      ctx->SetOutputDim("ViterbiPath", {emission_dims[0], emission_dims[1]});
    } else {
      ctx->SetOutputDim("ViterbiPath", {emission_dims[0], 1});
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(
        OperatorWithKernel::IndicateVarDataType(ctx, "Emission"),
        platform::CPUPlace());
  }
};
}  // namespace phi

PD_REGISTER_KERNEL(
    crf_decoding, CPU, ALL_LAYOUT, phi::CRFDecodingOpKernel, float, double) {}
