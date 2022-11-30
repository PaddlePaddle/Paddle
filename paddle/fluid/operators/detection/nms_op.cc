/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class NMSOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Boxes",
             "(Tensor) "
             "Boxes is a Tensor with shape [N, 4] "
             "N is the number of boxes "
             "in last dimension in format [x1, x2, y1, y2] "
             "the relation should be ``0 <= x1 < x2 && 0 <= y1 < y2``.");

    AddOutput("KeepBoxesIdxs",
              "(Tensor) "
              "KeepBoxesIdxs is a Tensor with shape [N] ");
    AddAttr<float>(
        "iou_threshold",
        "iou_threshold is a threshold value used to compress similar boxes "
        "boxes with IoU > iou_threshold will be considered as overlapping "
        "and just one of them can be kept.")
        .SetDefault(1.0f)
        .AddCustomChecker([](const float& iou_threshold) {
          PADDLE_ENFORCE_LE(iou_threshold,
                            1.0f,
                            platform::errors::InvalidArgument(
                                "iou_threshold should less equal than 1.0 "
                                "but got %f",
                                iou_threshold));
          PADDLE_ENFORCE_GE(iou_threshold,
                            0.0f,
                            platform::errors::InvalidArgument(
                                "iou_threshold should greater equal than 0.0 "
                                "but got %f",
                                iou_threshold));
        });
    AddComment(R"DOC(
                NMS Operator.
                This Operator is used to perform Non-Maximum Compress for input boxes.
                Indices of boxes kept by NMS will be sorted by scores and output.
            )DOC");
  }
};

class NMSOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Boxes"), ctx.GetPlace());
  }
};

template <typename T>
class NMSKernel : public framework::OpKernel<T> {};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(nms,
                            NMSInferMetaFunctor,
                            PD_INFER_META(phi::NMSInferMeta));

REGISTER_OPERATOR(
    nms,
    ops::NMSOp,
    ops::NMSOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    NMSInferMetaFunctor);
