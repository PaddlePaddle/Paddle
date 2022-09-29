/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/funcs/detection/nms_util.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;

class MatrixNMSOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Scores"),
        platform::CPUPlace());
  }
};

class MatrixNMSOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("BBoxes",
             "(Tensor) A 3-D Tensor with shape "
             "[N, M, 4] represents the predicted locations of M bounding boxes"
             ", N is the batch size. "
             "Each bounding box has four coordinate values and the layout is "
             "[xmin, ymin, xmax, ymax], when box size equals to 4.");
    AddInput("Scores",
             "(Tensor) A 3-D Tensor with shape [N, C, M] represents the "
             "predicted confidence predictions. N is the batch size, C is the "
             "class number, M is number of bounding boxes. For each category "
             "there are total M scores which corresponding M bounding boxes. "
             " Please note, M is equal to the 2nd dimension of BBoxes. ");
    AddAttr<int>(
        "background_label",
        "(int, default: 0) "
        "The index of background label, the background label will be ignored. "
        "If set to -1, then all categories will be considered.")
        .SetDefault(0);
    AddAttr<float>("score_threshold",
                   "(float) "
                   "Threshold to filter out bounding boxes with low "
                   "confidence score.");
    AddAttr<float>("post_threshold",
                   "(float, default 0.) "
                   "Threshold to filter out bounding boxes with low "
                   "confidence score AFTER decaying.")
        .SetDefault(0.);
    AddAttr<int>("nms_top_k",
                 "(int64_t) "
                 "Maximum number of detections to be kept according to the "
                 "confidences after the filtering detections based on "
                 "score_threshold");
    AddAttr<int>("keep_top_k",
                 "(int64_t) "
                 "Number of total bboxes to be kept per image after NMS "
                 "step. -1 means keeping all bboxes after NMS step.");
    AddAttr<bool>("normalized",
                  "(bool, default true) "
                  "Whether detections are normalized.")
        .SetDefault(true);
    AddAttr<bool>("use_gaussian",
                  "(bool, default false) "
                  "Whether to use Gaussian as decreasing function.")
        .SetDefault(false);
    AddAttr<float>("gaussian_sigma",
                   "(float) "
                   "Sigma for Gaussian decreasing function, only takes effect "
                   "when 'use_gaussian' is enabled.")
        .SetDefault(2.);
    AddOutput("Out",
              "(LoDTensor) A 2-D LoDTensor with shape [No, 6] represents the "
              "detections. Each row has 6 values: "
              "[label, confidence, xmin, ymin, xmax, ymax]. "
              "the offsets in first dimension are called LoD, the number of "
              "offset is N + 1, if LoD[i + 1] - LoD[i] == 0, means there is "
              "no detected bbox.");
    AddOutput("Index",
              "(LoDTensor) A 2-D LoDTensor with shape [No, 1] represents the "
              "index of selected bbox. The index is the absolute index cross "
              "batches.");
    AddOutput("RoisNum", "(Tensor), Number of RoIs in each images.")
        .AsDispensable();
    AddComment(R"DOC(
This operator does multi-class matrix non maximum suppression (NMS) on batched
boxes and scores.
In the NMS step, this operator greedily selects a subset of detection bounding
boxes that have high scores larger than score_threshold, if providing this
threshold, then selects the largest nms_top_k confidences scores if nms_top_k
is larger than -1. Then this operator decays boxes score according to the
Matrix NMS scheme.
Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
per image if keep_top_k is larger than -1.
This operator support multi-class and batched inputs. It applying NMS
independently for each class. The outputs is a 2-D LoDTenosr, for each
image, the offsets in first dimension of LoDTensor are called LoD, the number
of offset is N + 1, where N is the batch size. If LoD[i + 1] - LoD[i] == 0,
means there is no detected bbox for this image. Now this operator has one more
output, which is RoisNum. The size of RoisNum is N, RoisNum[i] means the number of
detected bbox for this image.

For more information on Matrix NMS, please refer to:
https://arxiv.org/abs/2003.10152
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(matrix_nms,
                            MatrixNMSInferShapeFunctor,
                            PD_INFER_META(phi::MatrixNMSInferMeta));

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    matrix_nms,
    ops::MatrixNMSOp,
    ops::MatrixNMSOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    MatrixNMSInferShapeFunctor);

REGISTER_OP_VERSION(matrix_nms)
    .AddCheckpoint(R"ROC(Upgrade matrix_nms: add a new output [RoisNum].)ROC",
                   paddle::framework::compatible::OpVersionDesc().NewOutput(
                       "RoisNum", "The number of RoIs in each image."));
