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
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

framework::OpKernelType QuantOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library_ = framework::LibraryType::kMKLDNN;
  framework::DataLayout layout_ = framework::DataLayout::kMKLDNN;

  return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                 ctx.GetPlace(), layout_, library_);
}

void QuantOpMaker::Make() {
  AddInput("Input", "input data");
  AddOutput("Output", "output data");
  AddAttr<bool>("is_negative_input",
                "(bool, default false) Only used in mkldnn INT8 kernel")
      .SetDefault(false);
  AddAttr<float>("Scale", "scale data").SetDefault({1.0f});
  AddComment(R"DOC(This op will quantize data from FP32 to INT8)DOC");
}

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(quantize, ops::QuantOp, ops::QuantOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
