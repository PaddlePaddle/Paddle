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

#include "paddle/fluid/operators/detection/nms_op.h"
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

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
          PADDLE_ENFORCE_LE(iou_threshold, 1.0f,
                            platform::errors::InvalidArgument(
                                "iou_threshold should less equal than 1.0 "
                                "but got %f",
                                iou_threshold));
          PADDLE_ENFORCE_GE(iou_threshold, 0.0f,
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
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Boxes"), "Input", "Boxes", "NMS");
    OP_INOUT_CHECK(ctx->HasOutput("KeepBoxesIdxs"), "Output", "KeepBoxesIdxs",
                   "NMS");

    auto boxes_dim = ctx->GetInputDim("Boxes");
    PADDLE_ENFORCE_EQ(boxes_dim.size(), 2,
                      platform::errors::InvalidArgument(
                          "The Input Boxes must be 2-dimention "
                          "whose shape must be [N, 4] "
                          "N is the number of boxes "
                          "in last dimension in format [x1, x2, y1, y2]. "));
    auto num_boxes = boxes_dim[0];

    ctx->SetOutputDim("KeepBoxesIdxs", {num_boxes});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Boxes"), ctx.GetPlace());
  }
};

template <typename T>
static void NMS(const T* boxes_data, int64_t* output_data, float threshold,
                int64_t num_boxes) {
  auto num_masks = CeilDivide(num_boxes, 64);
  std::vector<uint64_t> masks(num_masks, 0);

  for (int64_t i = 0; i < num_boxes; ++i) {
    if (masks[i / 64] & 1ULL << (i % 64)) continue;
    T box_1[4];
    for (int k = 0; k < 4; ++k) {
      box_1[k] = boxes_data[i * 4 + k];
    }
    for (int64_t j = i + 1; j < num_boxes; ++j) {
      if (masks[j / 64] & 1ULL << (j % 64)) continue;
      T box_2[4];
      for (int k = 0; k < 4; ++k) {
        box_2[k] = boxes_data[j * 4 + k];
      }
      bool is_overlap = CalculateIoU<T>(box_1, box_2, threshold);
      if (is_overlap) {
        masks[j / 64] |= 1ULL << (j % 64);
      }
    }
  }

  int64_t output_data_idx = 0;
  for (int64_t i = 0; i < num_boxes; ++i) {
    if (masks[i / 64] & 1ULL << (i % 64)) continue;
    output_data[output_data_idx++] = i;
  }

  for (; output_data_idx < num_boxes; ++output_data_idx) {
    output_data[output_data_idx] = 0;
  }
}

template <typename T>
class NMSKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* boxes = context.Input<Tensor>("Boxes");
    Tensor* output = context.Output<Tensor>("KeepBoxesIdxs");
    int64_t* output_data = output->mutable_data<int64_t>(context.GetPlace());
    auto threshold = context.template Attr<float>("iou_threshold");
    NMS<T>(boxes->data<T>(), output_data, threshold, boxes->dims()[0]);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    nms, ops::NMSOp, ops::NMSOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(nms, ops::NMSKernel<float>, ops::NMSKernel<double>);
