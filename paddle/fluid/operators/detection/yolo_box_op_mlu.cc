// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {
template <typename T>
class YoloBoxMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* img_size = ctx.Input<phi::DenseTensor>("ImgSize");
    auto* boxes = ctx.Output<phi::DenseTensor>("Boxes");
    auto* scores = ctx.Output<phi::DenseTensor>("Scores");
    const std::vector<int> anchors = ctx.Attr<std::vector<int>>("anchors");
    auto class_num = ctx.Attr<int>("class_num");
    auto conf_thresh = ctx.Attr<float>("conf_thresh");
    auto downsample_ratio = ctx.Attr<int>("downsample_ratio");
    auto clip_bbox = ctx.Attr<bool>("clip_bbox");
    auto scale = ctx.Attr<float>("scale_x_y");
    auto iou_aware = ctx.Attr<bool>("iou_aware");
    auto iou_aware_factor = ctx.Attr<float>("iou_aware_factor");

    int anchor_num = anchors.size() / 2;
    int64_t size = anchors.size();
    auto dim_x = x->dims();
    int n = dim_x[0];
    int s = anchor_num;
    int h = dim_x[2];
    int w = dim_x[3];

    // The output of mluOpYoloBox: A 4-D tensor with shape [N, anchor_num, 4,
    // H*W], the coordinates of boxes, and a 4-D tensor with shape [N,
    // anchor_num, :attr:`class_num`, H*W], the classification scores of boxes.
    std::vector<int64_t> boxes_dim_mluops({n, s, 4, h * w});
    std::vector<int64_t> scores_dim_mluops({n, s, class_num, h * w});

    // In Paddle framework: A 3-D tensor with shape [N, M, 4], the coordinates
    // of boxes, and a 3-D tensor with shape [N, M, :attr:`class_num`], the
    // classification scores of boxes.
    std::vector<int64_t> boxes_out_dim({n, s, h * w, 4});
    std::vector<int64_t> scores_out_dim({n, s, h * w, class_num});

    auto& dev_ctx = ctx.template device_context<MLUDeviceContext>();
    phi::DenseTensor boxes_tensor_mluops =
        ctx.AllocateTmpTensor<T, MLUDeviceContext>({n, s, 4, h * w}, dev_ctx);
    phi::DenseTensor scores_tensor_mluops =
        ctx.AllocateTmpTensor<T, MLUDeviceContext>({n, s, class_num, h * w},
                                                   dev_ctx);
    MLUOpTensorDesc boxes_trans_desc_mluops(
        4, boxes_dim_mluops.data(), ToMluOpDataType<T>());
    MLUCnnlTensorDesc boxes_trans_desc_cnnl(
        4, boxes_dim_mluops.data(), ToCnnlDataType<T>());
    MLUOpTensorDesc scores_trans_desc_mluops(
        4, scores_dim_mluops.data(), ToMluOpDataType<T>());
    MLUCnnlTensorDesc scores_trans_desc_cnnl(
        4, scores_dim_mluops.data(), ToCnnlDataType<T>());

    boxes->mutable_data<T>(ctx.GetPlace());
    scores->mutable_data<T>(ctx.GetPlace());
    FillMLUTensorWithHostValue(ctx, static_cast<T>(0), boxes);
    FillMLUTensorWithHostValue(ctx, static_cast<T>(0), scores);

    MLUOpTensorDesc x_desc(*x, MLUOP_LAYOUT_ARRAY, ToMluOpDataType<T>());
    MLUOpTensorDesc img_size_desc(
        *img_size, MLUOP_LAYOUT_ARRAY, ToMluOpDataType<int32_t>());
    phi::DenseTensor anchors_temp(framework::TransToPhiDataType(VT::INT32));
    anchors_temp.Resize({size});
    paddle::framework::TensorFromVector(
        anchors, ctx.device_context(), &anchors_temp);
    MLUOpTensorDesc anchors_desc(anchors_temp);
    MLUCnnlTensorDesc boxes_desc_cnnl(
        4, boxes_out_dim.data(), ToCnnlDataType<T>());
    MLUCnnlTensorDesc scores_desc_cnnl(
        4, scores_out_dim.data(), ToCnnlDataType<T>());

    MLUOP::OpYoloBox(ctx,
                     x_desc.get(),
                     GetBasePtr(x),
                     img_size_desc.get(),
                     GetBasePtr(img_size),
                     anchors_desc.get(),
                     GetBasePtr(&anchors_temp),
                     class_num,
                     conf_thresh,
                     downsample_ratio,
                     clip_bbox,
                     scale,
                     iou_aware,
                     iou_aware_factor,
                     boxes_trans_desc_mluops.get(),
                     GetBasePtr(&boxes_tensor_mluops),
                     scores_trans_desc_mluops.get(),
                     GetBasePtr(&scores_tensor_mluops));
    const std::vector<int> perm = {0, 1, 3, 2};

    // transpose the boxes from [N, S, 4, H*W] to [N, S, H*W, 4]
    MLUCnnl::Transpose(ctx,
                       perm,
                       4,
                       boxes_trans_desc_cnnl.get(),
                       GetBasePtr(&boxes_tensor_mluops),
                       boxes_desc_cnnl.get(),
                       GetBasePtr(boxes));

    // transpose the scores from [N, S, class_num, H*W] to [N, S, H*W,
    // class_num]
    MLUCnnl::Transpose(ctx,
                       perm,
                       4,
                       scores_trans_desc_cnnl.get(),
                       GetBasePtr(&scores_tensor_mluops),
                       scores_desc_cnnl.get(),
                       GetBasePtr(scores));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(yolo_box, ops::YoloBoxMLUKernel<float>);
