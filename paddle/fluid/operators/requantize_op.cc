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

#include "paddle/fluid/operators/requantize_op.h"

#include "paddle/fluid/framework/op_version_registry.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

framework::OpKernelType ReQuantOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
      ctx.GetPlace(),
      framework::DataLayout::kMKLDNN,
      framework::LibraryType::kMKLDNN);
}

void ReQuantOpMaker::Make() {
  AddInput("Input", "Input data");
  AddOutput("Output", "Output data");
  AddAttr<float>("Scale_in", "Scale in data").SetDefault({1.0f});
  AddAttr<float>("Scale_out", "Scale out data").SetDefault({1.0f});
  AddAttr<float>("Shift_in", "Shift in data").SetDefault({1.0f});
  AddAttr<float>("Shift_out", "Shift out data").SetDefault({1.0f});
  AddComment(
      R"DOC(This op will re-quantize data from INT8 with scale_in to INT8 with scale_out)DOC");
}

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(requantize, ops::ReQuantOp, ops::ReQuantOpMaker);

REGISTER_OP_VERSION(requantize)
    .AddCheckpoint(
        R"ROC( Add new attributes [Shift_in, Shift_out])ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewAttr("Shift_in",
                     "Provide quantization shift value for input data",
                     1.0f)
            .NewAttr("Shift_out",
                     "Provide quantization shift value for output data",
                     1.0f));
