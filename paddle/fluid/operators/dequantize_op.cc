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

#include "paddle/fluid/operators/dequantize_op.h"

#include "paddle/fluid/framework/op_version_registry.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

framework::OpKernelType DeQuantOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  auto input_data_type =
      framework::OperatorWithKernel::IndicateVarDataType(ctx, "Input");

  return framework::OpKernelType(input_data_type,
                                 ctx.GetPlace(),
                                 framework::DataLayout::kMKLDNN,
                                 framework::LibraryType::kMKLDNN);
}

void DeQuantOpMaker::Make() {
  AddInput("Input", "Input data");
  AddOutput("Output", "Output data");
  AddAttr<float>("Scale", "Scale data").SetDefault({1.0f});
  AddAttr<float>("Shift", "Shift data").SetDefault({0.0f});
  AddComment(R"DOC(This op will dequantize data from INT8 to FP32)DOC");
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(dequantize, ops::DeQuantOp, ops::DeQuantOpMaker);

REGISTER_OP_VERSION(dequantize)
    .AddCheckpoint(R"ROC( Add a new attribute [Shift])ROC",
                   paddle::framework::compatible::OpVersionDesc().NewAttr(
                       "Shift",
                       "Dequantize data to uint8 if provided non-zero value.",
                       0.0f));
