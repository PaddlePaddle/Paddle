/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/operators/quantize_op.h"

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

framework::OpKernelType QuantOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
      ctx.GetPlace(),
      phi::DataLayout::ONEDNN,
      framework::LibraryType::kMKLDNN);
}

void QuantOpMaker::Make() {
  AddInput("Input", "Input data");
  AddOutput("Output", "Output data");
  AddAttr<bool>("is_negative_input",
                "(bool, default false) Only used in mkldnn INT8 kernel")
      .SetDefault(false);
  AddAttr<float>("Scale", "Scale data").SetDefault({1.0f});
  AddAttr<float>(
      "Shift",
      "Shift data. When Shift is non-zero, data is quantized to unsigned int8.")
      .SetDefault({0.0f});
  AddAttr<std::string>("output_format",
                       "Convert format to NHWC or NCHW during quantization.")
      .SetDefault("NHWC");
  AddAttr<bool>("bfloat16", "(bool, default false) Convert to bfloat16")
      .SetDefault(false);
  AddComment(R"DOC(This op will quantize data from FP32 to INT8)DOC");
}

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(quantize, ops::QuantOp, ops::QuantOpMaker);

REGISTER_OP_VERSION(quantize)
    .AddCheckpoint(R"ROC( Add a new attribute [bfloat16])ROC",
                   paddle::framework::compatible::OpVersionDesc().NewAttr(
                       "bfloat16",
                       "If true, float32 input is converted to bfloat16",
                       false))
    .AddCheckpoint(R"ROC( Add a new attribute [Shift])ROC",
                   paddle::framework::compatible::OpVersionDesc().NewAttr(
                       "Shift",
                       "Quantize data to uint8 if provided non-zero value.",
                       0.0f));
