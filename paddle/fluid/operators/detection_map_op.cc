/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection_map_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class DetectionMAPOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Detection"),
                   "Input(Detection) of DetectionMAPOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"),
                   "Input(Label) of DetectionMAPOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("OutPosCount"),
                   "Output(OutPosCount) of DetectionMAPOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("OutTruePos"),
                   "Output(OutTruePos) of DetectionMAPOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("OutFalsePos"),
                   "Output(OutFalsePos) of DetectionMAPOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MAP"),
                   "Output(MAP) of DetectionMAPOp should not be null.");

    auto det_dims = ctx->GetInputDim("Detection");
    PADDLE_ENFORCE_EQ(det_dims.size(), 2UL,
                      "The rank of Input(Detection) must be 2, "
                      "the shape is [N, 6].");
    PADDLE_ENFORCE_EQ(det_dims[1], 6UL,
                      "The shape is of Input(Detection) [N, 6].");
    auto label_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE_EQ(label_dims.size(), 2UL,
                      "The rank of Input(Label) must be 2, "
                      "the shape is [N, 6].");
    PADDLE_ENFORCE_EQ(label_dims[1], 6UL,
                      "The shape is of Input(Label) [N, 6].");

    auto map_dim = framework::make_ddim({1});
    ctx->SetOutputDim("MAP", map_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(
            ctx.Input<framework::Tensor>("Detection")->type()),
        ctx.device_context());
  }
};

class DetectionMAPOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  DetectionMAPOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Label",
             "(LoDTensor) A 2-D LoDTensor with shape[N, 6] represents the"
             "Labeled ground-truth data. Each row has 6 values: "
             "[label, is_difficult, xmin, ymin, xmax, ymax], N is the total "
             "number of ground-truth data in this mini-batch. For each "
             "instance, the offsets in first dimension are called LoD, "
             "the number of offset is N + 1, if LoD[i + 1] - LoD[i] == 0, "
             "means there is no ground-truth data.");
    AddInput("Detection",
             "(LoDTensor) A 2-D LoDTensor with shape [M, 6] represents the "
             "detections. Each row has 6 values: "
             "[label, confidence, xmin, ymin, xmax, ymax], M is the total "
             "number of detections in this mini-batch. For each instance, "
             "the offsets in first dimension are called LoD, the number of "
             "offset is N + 1, if LoD[i + 1] - LoD[i] == 0, means there is "
             "no detected data.");
    AddInput("PosCount",
             "(Tensor) A tensor with shape [Ncls, 1], store the "
             "input positive example count of each class.")
        .AsDispensable();
    AddInput("TruePos",
             "(LodTensor) A 2-D LodTensor with shape [Ntp, 2], store the "
             "input true positive example of each class.")
        .AsDispensable();
    AddInput("FalsePos",
             "(LodTensor) A 2-D LodTensor with shape [Nfp, 2], store the "
             "input false positive example of each class.")
        .AsDispensable();
    AddOutput("OutPosCount",
              "(Tensor) A tensor with shape [Ncls, 1], store the "
              "positive example count of each class. It combines the input "
              "input(PosCount) and the positive example count computed from "
              "input(Detection) and input(Label).");
    AddOutput("OutTruePos",
              "(LodTensor) A LodTensor with shape [Ntp', 2], store the "
              "true positive example of each class. It combines the "
              "input(TruePos) and the true positive examples computed from "
              "input(Detection) and input(Label).");
    AddOutput("OutFalsePos",
              "(LodTensor) A LodTensor with shape [Nfp', 2], store the "
              "false positive example of each class. It combines the "
              "input(FalsePos) and the false positive examples computed from "
              "input(Detection) and input(Label).");
    AddOutput("MAP",
              "(Tensor) A tensor with shape [1], store the mAP evaluate "
              "result of the detection.");

    AddAttr<float>("overlap_threshold",
                   "(float) "
                   "The jaccard overlap threshold of detection output and "
                   "ground-truth data.")
        .SetDefault(.3f);
    AddAttr<bool>("evaluate_difficult",
                  "(bool, default true) "
                  "Switch to control whether the difficult data is evaluated.")
        .SetDefault(true);
    AddAttr<std::string>("ap_type",
                         "(string, default 'integral') "
                         "The AP algorithm type, 'integral' or '11point'.")
        .SetDefault("integral")
        .InEnum({"integral", "11point"})
        .AddCustomChecker([](const std::string& ap_type) {
          PADDLE_ENFORCE_NE(GetAPType(ap_type), APType::kNone,
                            "The ap_type should be 'integral' or '11point.");
        });
    AddComment(R"DOC(
Detection mAP evaluate operator.
The general steps are as follows. First, calculate the true positive and
 false positive according to the input of detection and labels, then
 calculate the mAP evaluate value.
 Supporting '11 point' and 'integral' mAP algorithm. Please get more information
 from the following articles:
 https://sanchom.wordpress.com/tag/average-precision/
 https://arxiv.org/abs/1512.02325

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(detection_map, ops::DetectionMAPOp,
                             ops::DetectionMAPOpMaker);
REGISTER_OP_CPU_KERNEL(
    detection_map, ops::DetectionMAPOpKernel<paddle::platform::CPUPlace, float>,
    ops::DetectionMAPOpKernel<paddle::platform::CPUPlace, double>);
