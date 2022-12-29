/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/funcs/detection/nms_util.h"
#include "paddle/phi/kernels/funcs/gather.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

class GenerateProposalsV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Anchors"),
        ctx.device_context());
  }
};

class GenerateProposalsV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Scores",
             "(Tensor) The scores from conv is in shape (N, A, H, W), "
             "N is batch size, A is number of anchors, "
             "H and W are height and width of the feature map");
    AddInput("BboxDeltas",
             "(Tensor) Bounding box deltas from conv is in "
             "shape (N, 4*A, H, W).");
    AddInput("ImShape",
             "(Tensor) Image shape in shape (N, 2), "
             "in format (height, width)");
    AddInput("Anchors",
             "(Tensor) Bounding box anchors from anchor_generator_op "
             "is in shape (A, H, W, 4).");
    AddInput("Variances",
             "(Tensor) Bounding box variances with same shape as `Anchors`.");

    AddOutput("RpnRois",
              "(phi::DenseTensor), Output proposals with shape (rois_num, 4).");
    AddOutput(
        "RpnRoiProbs",
        "(phi::DenseTensor) Scores of proposals with shape (rois_num, 1).");
    AddOutput("RpnRoisNum", "(Tensor), The number of Rpn RoIs in each image")
        .AsDispensable();
    AddAttr<int>("pre_nms_topN",
                 "Number of top scoring RPN proposals to keep before "
                 "applying NMS.");
    AddAttr<int>("post_nms_topN",
                 "Number of top scoring RPN proposals to keep after "
                 "applying NMS");
    AddAttr<float>("nms_thresh", "NMS threshold used on RPN proposals.");
    AddAttr<float>("min_size",
                   "Proposal height and width both need to be greater "
                   "than this min_size.");
    AddAttr<float>("eta", "The parameter for adaptive NMS.");
    AddAttr<bool>("pixel_offset",
                  "(bool, default True),"
                  "If true, im_shape pixel offset is 1.")
        .SetDefault(true);
    AddComment(R"DOC(
This operator is the second version of generate_proposals op to generate
bounding box proposals for Faster RCNN.
The proposals are generated for a list of images based on image
score 'Scores', bounding box regression result 'BboxDeltas' as
well as predefined bounding box shapes 'anchors'. Greedy
non-maximum suppression is applied to generate the final bounding
boxes.

The difference between this version and the first version is that the image
 scale is no long needed now, so the input requires im_shape instead of im_info.
The change aims to unify the input for all kinds of objective detection
such as YOLO-v3 and Faster R-CNN. As a result, the min_size represents the
size on input image instead of original image which is slightly different
to before and will not effect the result.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(generate_proposals_v2,
                            GenerateProposalsV2InferShapeFunctor,
                            PD_INFER_META(phi::GenerateProposalsV2InferMeta));

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    generate_proposals_v2,
    ops::GenerateProposalsV2Op,
    ops::GenerateProposalsV2OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    GenerateProposalsV2InferShapeFunctor);

REGISTER_OP_VERSION(generate_proposals_v2)
    .AddCheckpoint(
        R"ROC(Registe generate_proposals_v2 for adding the attribute of pixel_offset)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "pixel_offset", "If true, im_shape pixel offset is 1.", true));
