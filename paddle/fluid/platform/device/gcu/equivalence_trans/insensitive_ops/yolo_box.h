/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"
#include "paddle/phi/kernels/funcs/yolo_box_util.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kYoloBox = "yolo_box";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, YoloBoxEquivalenceTrans) {
  auto input = *(map_inputs["X"].at(0));
  auto img_size = *(map_inputs["ImgSize"].at(0));
  auto *op = node->Op();
  auto class_num = PADDLE_GET_CONST(int, op->GetAttr("class_num"));
  auto anchors = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("anchors"));
  auto downsample_ratio =
      PADDLE_GET_CONST(int, op->GetAttr("downsample_ratio"));
  auto conf_thresh = PADDLE_GET_CONST(float, op->GetAttr("conf_thresh"));
  auto clip_bbox = PADDLE_GET_CONST(bool, op->GetAttr("clip_bbox"));
  auto scale_x_y = PADDLE_GET_CONST(float, op->GetAttr("scale_x_y"));
  auto iou_aware = PADDLE_GET_CONST(bool, op->GetAttr("iou_aware"));
  auto iou_aware_factor =
      PADDLE_GET_CONST(float, op->GetAttr("iou_aware_factor"));

  const auto input_shape = input.GetType().GetShape();
  if (input.IsDynamic()) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Not support dynamic input[%s] for yolo_box in gcu backend.",
        phi::make_ddim(input_shape).to_str()));
  }

  const int64_t n = input_shape[0];
  const int64_t h = input_shape[2];
  const int64_t w = input_shape[3];
  const int64_t anchor_num = anchors.size() / 2;
  const int64_t box_num = h * w * anchor_num;
  const int64_t input_size_h = downsample_ratio * h;
  const int64_t input_size_w = downsample_ratio * w;
  const int64_t stride = h * w;
  const int64_t an_stride = (class_num + 5) * stride;
  const float bias = -0.5 * (scale_x_y - 1.0);

  // input, output indices
  std::vector<int64_t> input_indices;
  std::vector<int64_t> iou_indices;
  std::vector<int64_t> img_height_indices;
  std::vector<int64_t> img_width_indices;
  std::vector<int64_t> box_indices;
  std::vector<int64_t> label_indices;
  std::vector<int64_t> score_indices;

  // const data
  std::vector<int64_t> box_0_i;
  std::vector<int64_t> box_1_j;
  std::vector<int64_t> box_2_anchors;
  std::vector<int64_t> box_3_anchors;

  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < anchor_num; ++j) {
      for (int64_t k = 0; k < h; ++k) {
        for (int64_t l = 0; l < w; ++l) {
          img_height_indices.emplace_back(2 * i);
          img_width_indices.emplace_back(2 * i + 1);

          int64_t obj_idx = phi::funcs::GetEntryIndex(
              i, j, k * w + l, anchor_num, an_stride, stride, 4, iou_aware);
          input_indices.emplace_back(obj_idx);

          if (iou_aware) {
            int64_t iou_idx = phi::funcs::GetIoUIndex(
                i, j, k * w + l, anchor_num, an_stride, stride);
            iou_indices.emplace_back(iou_idx);
          }

          int64_t box_idx = phi::funcs::GetEntryIndex(
              i, j, k * w + l, anchor_num, an_stride, stride, 0, iou_aware);
          box_indices.emplace_back(box_idx);

          box_0_i.emplace_back(l);
          box_1_j.emplace_back(k);
          box_2_anchors.emplace_back(anchors[2 * j]);
          box_3_anchors.emplace_back(anchors[2 * j + 1]);

          int64_t label_idx = phi::funcs::GetEntryIndex(
              i, j, k * w + l, anchor_num, an_stride, stride, 5, iou_aware);
          label_indices.emplace_back(label_idx);

          int64_t score_idx =
              (i * box_num + j * stride + k * w + l) * class_num;
          score_indices.emplace_back(score_idx);
        }
      }
    }
  }

  auto input_dtype = input.GetType().GetPrimitiveType();
  input = builder::FlattenV2(input);
  auto sigmoid_input = builder::Sigmoid(input);
  auto exp_input = builder::Exp(input);

  auto gather_func =
      [](GcuOp input, const std::vector<int64_t> &indices, int64_t offset = 0) {
        std::vector<int64_t> new_indices;
        for (auto index : indices) new_indices.emplace_back(index + offset);

        auto const_indices = builder::Const(
            input.GetBuilder(),
            new_indices,
            builder::Type({static_cast<int64_t>(new_indices.size())},
                          builder::PrimitiveType::S64()));

        std::vector<int64_t> offset_dims;
        int64_t axis = 0;
        int64_t index_vector_dim = 1;
        auto dnums =
            builder::GatherDimensionNumbers(offset_dims,
                                            /*collapsed_slice_dims*/ {axis},
                                            /*start_index_map*/ {axis},
                                            index_vector_dim);

        std::vector<int64_t> slice_sizes(input.GetType().GetShape());
        slice_sizes[axis] = 1;
        return builder::Gather(input, const_indices, dnums, slice_sizes, false);
      };

  auto indice_input = gather_func(input, input_indices);
  auto indice_sigmoid_input = gather_func(sigmoid_input, input_indices);

  GcuOp conf;
  GcuOp cond;
  if (!iou_aware) {
    conf = indice_sigmoid_input;
  } else {
    auto iou = gather_func(sigmoid_input, iou_indices);
    auto tmp1 = builder::Const(gcu_builder,
                               1.0 - iou_aware_factor,
                               builder::Type(builder::PrimitiveType::F32()));
    auto tmp2 = builder::Const(gcu_builder,
                               iou_aware_factor,
                               builder::Type(builder::PrimitiveType::F32()));
    conf = builder::Pow(indice_sigmoid_input, tmp1) * builder::Pow(iou, tmp2);
  }
  cond = conf >= builder::FullLike(conf, conf_thresh);

  img_size = builder::FlattenV2(img_size);
  auto img_height = gather_func(img_size, img_height_indices);
  auto img_width = gather_func(img_size, img_width_indices);
  img_height = builder::Convert(
      img_height, builder::Type(img_height.GetType().GetShape(), input_dtype));
  img_width = builder::Convert(
      img_width, builder::Type(img_width.GetType().GetShape(), input_dtype));

  auto box_0_input = gather_func(sigmoid_input, box_indices);
  auto box_1_input = gather_func(sigmoid_input, box_indices, stride);
  auto box_2_input = gather_func(exp_input, box_indices, 2 * stride);
  auto box_3_input = gather_func(exp_input, box_indices, 3 * stride);

  auto const_scale = builder::FullLike(indice_input, scale_x_y);
  auto const_bias = builder::FullLike(indice_input, bias);
  auto const_grid_size_w = builder::FullLike(indice_input, w);
  auto const_grid_size_h = builder::FullLike(indice_input, h);
  auto const_input_size_w = builder::FullLike(indice_input, input_size_w);
  auto const_input_size_h = builder::FullLike(indice_input, input_size_h);

  auto const_box_0_i = builder::Const(
      gcu_builder,
      box_0_i,
      builder::Type({static_cast<int64_t>(box_0_i.size())}, input_dtype));
  auto const_box_1_j = builder::Const(
      gcu_builder,
      box_1_j,
      builder::Type({static_cast<int64_t>(box_1_j.size())}, input_dtype));
  auto const_box_2_anchors = builder::Const(
      gcu_builder,
      box_2_anchors,
      builder::Type({static_cast<int64_t>(box_2_anchors.size())}, input_dtype));
  auto const_box_3_anchors = builder::Const(
      gcu_builder,
      box_3_anchors,
      builder::Type({static_cast<int64_t>(box_3_anchors.size())}, input_dtype));

  auto box_0 = (const_box_0_i + box_0_input * const_scale + const_bias) *
               img_width / const_grid_size_w;
  auto box_1 = (const_box_1_j + box_1_input * const_scale + const_bias) *
               img_height / const_grid_size_h;
  auto box_2 =
      box_2_input * const_box_2_anchors * img_width / const_input_size_w;
  auto box_3 =
      box_3_input * const_box_3_anchors * img_height / const_input_size_h;

  auto const_2 = builder::FullLike(indice_input, 2);
  auto boxes_0 = box_0 - box_2 / const_2;
  auto boxes_1 = box_1 - box_3 / const_2;
  auto boxes_2 = box_0 + box_2 / const_2;
  auto boxes_3 = box_1 + box_3 / const_2;

  if (clip_bbox) {
    auto const_0 = builder::ZerosLike(indice_input);
    auto img_w_sub_1 = img_width - builder::OnesLike(img_width);
    auto img_h_sub_1 = img_height - builder::OnesLike(img_height);
    boxes_0 = builder::Select(boxes_0 > const_0, boxes_0, const_0);
    boxes_1 = builder::Select(boxes_1 > const_0, boxes_1, const_0);
    boxes_2 = builder::Select(boxes_2 < img_w_sub_1, boxes_2, img_w_sub_1);
    boxes_3 = builder::Select(boxes_3 < img_h_sub_1, boxes_3, img_h_sub_1);
  }
  std::vector<int64_t> new_shape = {
      static_cast<int64_t>(indice_input.GetType().GetSize()), 1};
  boxes_0 = builder::Reshape(boxes_0, new_shape);
  boxes_1 = builder::Reshape(boxes_1, new_shape);
  boxes_2 = builder::Reshape(boxes_2, new_shape);
  boxes_3 = builder::Reshape(boxes_3, new_shape);
  auto box = builder::Concatenate({boxes_0, boxes_1, boxes_2, boxes_3}, 1);

  std::vector<GcuOp> ops;
  for (int64_t i = 0; i < class_num; ++i) {
    auto input_score = gather_func(sigmoid_input, label_indices, i * stride);
    auto scores = conf * input_score;
    ops.emplace_back(builder::Reshape(scores, new_shape));
  }
  auto score = builder::Concatenate(ops, 1);

  std::vector<int64_t> box_shape({n, box_num, 4});
  std::vector<int64_t> score_shape({n, box_num, class_num});

  auto box_size = box.GetType().GetSize();
  auto score_size = score.GetType().GetSize();
  PADDLE_ENFORCE_EQ(
      box_size,
      n * box_num * 4,
      platform::errors::InvalidArgument(
          "Output Box size[%d] should be %d", box_size, n * box_num * 4));
  PADDLE_ENFORCE_EQ(
      score_size,
      n * box_num * class_num,
      platform::errors::InvalidArgument("Output Box size[%d] should be %d",
                                        box_size,
                                        n * box_num * class_num));

  auto cond_box =
      builder::BroadcastInDim(cond,
                              {0},
                              builder::Type(box.GetType().GetShape(),
                                            cond.GetType().GetPrimitiveType()));
  auto cond_score =
      builder::BroadcastInDim(cond,
                              {0},
                              builder::Type(score.GetType().GetShape(),
                                            cond.GetType().GetPrimitiveType()));
  box = builder::Reshape(box, box_shape);
  score = builder::Reshape(score, score_shape);
  cond_box = builder::Reshape(cond_box, box_shape);
  cond_score = builder::Reshape(cond_score, score_shape);
  auto init_box = builder::ZerosLike(box);
  auto init_score = builder::ZerosLike(score);

  auto out_box = builder::Select(cond_box, box, init_box);

  auto out_score = builder::Select(cond_score, score, init_score);
  auto output = builder::Tuple({out_box, out_score});

  auto output_name_map = op->Outputs();
  std::string output_names_attr =
      output_name_map["Boxes"][0] + ";" + output_name_map["Scores"][0];
  output.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));

  return std::make_shared<GcuOp>(output);
}

EQUIVALENCE_TRANS_FUNC_REG(kYoloBox, INSENSITIVE, YoloBoxEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
