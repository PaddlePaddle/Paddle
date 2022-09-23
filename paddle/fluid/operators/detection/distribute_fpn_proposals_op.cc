/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class DistributeFpnProposalsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "FpnRois");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class DistributeFpnProposalsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("FpnRois", "(LoDTensor) The RoIs at all levels in shape (-1, 4)");
    AddInput("RoisNum",
             "(Tensor) The number of RoIs in shape (B),"
             "B is the number of images")
        .AsDispensable();
    AddOutput("MultiFpnRois", "(LoDTensor) Output with distribute operator")
        .AsDuplicable();
    AddOutput("RestoreIndex",
              "(Tensor) An array of positive number which is "
              "used to restore the order of FpnRois");
    AddOutput("MultiLevelRoIsNum",
              "(List of Tensor) The RoIs' number of each image on multiple "
              "levels. The number on each level has the shape of (B),"
              "B is the number of images.")
        .AsDuplicable()
        .AsDispensable();
    AddAttr<int>("min_level",
                 "The lowest level of FPN layer where the"
                 " proposals come from");
    AddAttr<int>("max_level",
                 "The highest level of FPN layer where the"
                 " proposals come from");
    AddAttr<int>("refer_level",
                 "The referring level of FPN layer with"
                 " specified scale");
    AddAttr<int>("refer_scale",
                 "The referring scale of FPN layer with"
                 " specified level");
    AddAttr<bool>("pixel_offset",
                  "(bool, default True),"
                  "If true, im_shape pixel offset is 1.")
        .SetDefault(true);
    AddComment(R"DOC(
This operator distribute all proposals into different fpn level,
 with respect to scale of the proposals, the referring scale and
 the referring level. Besides, to restore the order of proposals,
we return an array which indicate the original index of rois in
 current proposals.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(
    distribute_fpn_proposals,
    DistributeFpnProposalsInferShapeFunctor,
    PD_INFER_META(phi::DistributeFpnProposalsInferMeta));

REGISTER_OPERATOR(
    distribute_fpn_proposals,
    ops::DistributeFpnProposalsOp,
    ops::DistributeFpnProposalsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    DistributeFpnProposalsInferShapeFunctor);
REGISTER_OP_VERSION(distribute_fpn_proposals)
    .AddCheckpoint(
        R"ROC(
              Upgrade distribute_fpn_proposals add a new input
              [RoisNum] and add a new output [MultiLevelRoIsNum].)ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewInput("RoisNum", "The number of RoIs in each image.")
            .NewOutput("MultiLevelRoisNum",
                       "The RoIs' number of each image on multiple "
                       "levels. The number on each level has the shape of (B),"
                       "B is the number of images."))
    .AddCheckpoint(
        R"ROC(Register distribute_fpn_proposals for adding the attribute of pixel_offset)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "pixel_offset", "If true, im_shape pixel offset is 1.", true));
