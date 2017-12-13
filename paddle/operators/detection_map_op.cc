/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/detection_map_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class DetectionMAPOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto map_dim = framework::make_ddim({1});
    ctx->SetOutputDim("MAP", map_dim);
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("Label")->type()),
        ctx.device_context());
  }
};

class DetectionMAPOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  DetectionMAPOpMaker(framework::OpProto* proto,
                      framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Detect", "The detection output.");
    AddInput("Label", "The label data.");
    AddOutput("MAP", "The MAP evaluate result of the detection.");

    AddAttr<float>("overlap_threshold", "The overlap threshold.")
        .SetDefault(.3f);
    AddAttr<bool>("evaluate_difficult",
                  "Switch to control whether the difficult data is evaluated.")
        .SetDefault(true);
    AddAttr<std::string>("ap_type",
                         "The AP algorithm type, 'Integral' or '11point'.")
        .SetDefault("Integral");

    AddComment(R"DOC(
Detection MAP Operator.

Detection MAP evaluator for SSD(Single Shot MultiBox Detector) algorithm.
Please get more information from the following papers:
https://arxiv.org/abs/1512.02325.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(detection_map, ops::DetectionMAPOp,
                             ops::DetectionMAPOpMaker);
REGISTER_OP_CPU_KERNEL(
    detection_map, ops::DetectionMAPOpKernel<paddle::platform::GPUPlace, float>,
    ops::DetectionMAPOpKernel<paddle::platform::GPUPlace, double>);
