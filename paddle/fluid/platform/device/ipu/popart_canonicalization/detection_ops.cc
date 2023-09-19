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

#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {
namespace {

Node *yolo_box_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto clip_bbox = PADDLE_GET_CONST(bool, op->GetAttr("clip_bbox"));
  auto iou_aware = PADDLE_GET_CONST(bool, op->GetAttr("iou_aware"));
  auto conf_thresh = PADDLE_GET_CONST(float, op->GetAttr("conf_thresh"));
  auto iou_aware_factor =
      PADDLE_GET_CONST(float, op->GetAttr("iou_aware_factor"));
  auto class_num = PADDLE_GET_CONST(int, op->GetAttr("class_num"));
  auto downsample_ratio =
      PADDLE_GET_CONST(int, op->GetAttr("downsample_ratio"));
  auto scale_x_y = PADDLE_GET_CONST(float, op->GetAttr("scale_x_y"));
  auto anchors = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("anchors"));

  // For Slice Op, while value is very large, it equals to the ends.
  int max_int = INT_MAX;
  int anchor_num = anchors.size() / 2;

  // FP32 or FP16
  auto target_dtype = GetInputVarNode("X", node)->Var()->GetDataType();

  Node *input_x = GetInputVarNode("X", node);
  if (iou_aware) {
    input_x =
        CreateSlice(graph,
                    node,
                    {input_x},
                    {},
                    std::vector<int>{0, 0, 0, 0},
                    std::vector<int>{max_int, anchor_num, max_int, max_int},
                    std::vector<int>{0, 1, 2, 3},
                    std::vector<int>{1, 1, 1, 1})
            ->outputs[0];
  }
  auto nchw = GetInputVarNode("X", node)->Var()->GetShape();
  // Channel `C` = anchor_num * (5 + class_num)
  auto *reshaped_x =
      CreateReshape(
          graph,
          node,
          {input_x},
          {},
          std::vector<int64_t>{nchw[0], anchor_num, -1, nchw[2], nchw[3]})
          ->outputs[0];
  auto *transposed_x =
      CreateBaseOp(graph,
                   node,
                   "popart_transpose",
                   {reshaped_x},
                   {},
                   {{"perm", std::vector<int64_t>{0, 1, 3, 4, 2}}})
          ->outputs[0];

  // Build the grid
  // grid_x_0 shape is [w]
  std::vector<float> grid_x_0(nchw[3]);
  std::iota(grid_x_0.begin(), grid_x_0.end(), 0.0f);
  // grid_y_0 shape is [h]
  std::vector<float> grid_y_0(nchw[2]);
  std::iota(grid_y_0.begin(), grid_y_0.end(), 0.0f);
  // grid_x_1 shape is [w * h]
  std::vector<float> grid_x_1;
  for (int i = 0; i < nchw[2]; i++) {
    grid_x_1.insert(grid_x_1.end(), grid_x_0.begin(), grid_x_0.end());
  }
  auto *grid_x_1_node = CreateConst(graph,
                                    node,
                                    grid_x_1,
                                    {int64_t(grid_x_1.size())},
                                    VarType2OnnxDType(target_dtype))
                            ->outputs[0];
  // grid_y_1 shape is [h * w]
  std::vector<float> grid_y_1;
  for (int i = 0; i < nchw[3]; i++) {
    grid_y_1.insert(grid_y_1.end(), grid_y_0.begin(), grid_y_0.end());
  }
  auto *grid_y_1_node = CreateConst(graph,
                                    node,
                                    grid_y_1,
                                    {int64_t(grid_y_1.size())},
                                    VarType2OnnxDType(target_dtype))
                            ->outputs[0];
  auto *grid_x_node = CreateReshape(graph,
                                    node,
                                    {grid_x_1_node},
                                    {},
                                    std::vector<int64_t>{nchw[2], nchw[3], 1})
                          ->outputs[0];
  auto *grid_y_2_node = CreateReshape(graph,
                                      node,
                                      {grid_y_1_node},
                                      {},
                                      std::vector<int64_t>{nchw[3], nchw[2], 1})
                            ->outputs[0];
  auto *grid_y_node = CreateBaseOp(graph,
                                   node,
                                   "popart_transpose",
                                   {grid_y_2_node},
                                   {},
                                   {{"perm", std::vector<int64_t>{1, 0, 2}}})
                          ->outputs[0];
  auto *grid_node = CreateBaseOp(graph,
                                 node,
                                 "popart_concat",
                                 {grid_x_node, grid_y_node},
                                 {},
                                 {{"axis", int64_t(2)}})
                        ->outputs[0];

  // Generate the positions(x, y) of boxes
  // pred_box[:, :, :, :, 0] = (grid_x + sigmoid(pred_box[:, :, :, :, 0]) *
  // scale_x_y + bias_x_y) / w pred_box[:, :, :, :, 1] = (grid_y +
  // sigmoid(pred_box[:, :, :, :, 1]) * scale_x_y + bias_x_y) / h
  auto *pred_box_xy =
      CreateSlice(graph,
                  node,
                  {transposed_x},
                  {},
                  std::vector<int>{0, 0, 0, 0, 0},
                  std::vector<int>{max_int, max_int, max_int, max_int, 2},
                  std::vector<int>{0, 1, 2, 3, 4},
                  std::vector<int>{1, 1, 1, 1, 1})
          ->outputs[0];
  auto *scale_x_y_node = CreateConst(graph,
                                     node,
                                     std::vector<float>{scale_x_y},
                                     {int64_t(1)},
                                     VarType2OnnxDType(target_dtype))
                             ->outputs[0];
  auto *bias_x_y_node =
      CreateConst(graph,
                  node,
                  std::vector<float>{(1.0f - scale_x_y) / 2.0f},
                  {int64_t(1)},
                  VarType2OnnxDType(target_dtype))
          ->outputs[0];
  auto *wh = CreateConst(graph,
                         node,
                         std::vector<float>{static_cast<float>(nchw[3]),
                                            static_cast<float>(nchw[2])},
                         {int64_t(2)},
                         VarType2OnnxDType(target_dtype))
                 ->outputs[0];
  pred_box_xy = CreateBaseOp(graph, node, "popart_sigmoid", {pred_box_xy}, {})
                    ->outputs[0];
  pred_box_xy =
      CreateBaseOp(graph, node, "popart_mul", {pred_box_xy, scale_x_y_node}, {})
          ->outputs[0];
  pred_box_xy =
      CreateBaseOp(graph, node, "popart_add", {pred_box_xy, bias_x_y_node}, {})
          ->outputs[0];
  pred_box_xy =
      CreateBaseOp(graph, node, "popart_add", {pred_box_xy, grid_node}, {})
          ->outputs[0];
  pred_box_xy = CreateBaseOp(graph, node, "popart_div", {pred_box_xy, wh}, {})
                    ->outputs[0];

  // Generate Width and Height of boxes
  // anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
  // anchors_s = np.array(
  //     [(an_w / input_w, an_h / input_h) for an_w, an_h in anchors])
  // anchor_w = anchors_s[:, 0:1].reshape((1, an_num, 1, 1))
  // anchor_h = anchors_s[:, 1:2].reshape((1, an_num, 1, 1))
  auto *anchors_node =
      CreateConst(
          graph,
          node,
          std::vector<float>{anchors.begin(), anchors.begin() + anchor_num * 2},
          {int64_t(anchor_num * 2)},
          VarType2OnnxDType(target_dtype))
          ->outputs[0];
  anchors_node =
      CreateReshape(
          graph, node, {anchors_node}, {}, std::vector<int64_t>{anchor_num, 2})
          ->outputs[0];
  auto *downsample_node =
      CreateConst(graph,
                  node,
                  std::vector<float>{static_cast<float>(downsample_ratio)},
                  {int64_t(1)},
                  VarType2OnnxDType(target_dtype))
          ->outputs[0];
  auto *ori_wh =
      CreateBaseOp(graph, node, "popart_mul", {wh, downsample_node}, {})
          ->outputs[0];
  anchors_node =
      CreateBaseOp(graph, node, "popart_div", {anchors_node, ori_wh}, {})
          ->outputs[0];
  anchors_node = CreateReshape(graph,
                               node,
                               {anchors_node},
                               {},
                               std::vector<int64_t>{1, anchor_num, 1, 1, 2})
                     ->outputs[0];
  auto *pred_box_wh =
      CreateSlice(graph,
                  node,
                  {transposed_x},
                  {},
                  std::vector<int>{0, 0, 0, 0, 2},
                  std::vector<int>{max_int, max_int, max_int, max_int, 4},
                  std::vector<int>{0, 1, 2, 3, 4},
                  std::vector<int>{1, 1, 1, 1, 1})
          ->outputs[0];
  pred_box_wh =
      CreateBaseOp(graph, node, "popart_exp", {pred_box_wh}, {})->outputs[0];
  pred_box_wh =
      CreateBaseOp(graph, node, "popart_mul", {pred_box_wh, anchors_node}, {})
          ->outputs[0];

  // Ignore the boxes whose confidience lower than the threshold
  // if iou_aware:
  //     pred_conf = sigmoid(x[:, :, :, :, 4:5])**(
  //         1 - iou_aware_factor) * sigmoid(ioup)**iou_aware_factor
  // else:
  //     pred_conf = sigmoid(x[:, :, :, :, 4:5])
  auto *confidence =
      CreateSlice(graph,
                  node,
                  {transposed_x},
                  {},
                  std::vector<int>{0, 0, 0, 0, 4},
                  std::vector<int>{max_int, max_int, max_int, max_int, 5},
                  std::vector<int>{0, 1, 2, 3, 4},
                  std::vector<int>{1, 1, 1, 1, 1})
          ->outputs[0];
  auto *pred_conf =
      CreateBaseOp(graph, node, "popart_sigmoid", {confidence}, {})->outputs[0];
  if (iou_aware) {
    auto *ioup =
        CreateSlice(graph,
                    node,
                    {GetInputVarNode("X", node)},
                    {},
                    std::vector<int>{0, 0, 0, 0},
                    std::vector<int>{max_int, anchor_num, max_int, max_int},
                    std::vector<int>{0, 1, 2, 3},
                    std::vector<int>{1, 1, 1, 1})
            ->outputs[0];
    ioup = CreateBaseOp(graph,
                        node,
                        "popart_unsqueeze",
                        {ioup},
                        {},
                        {{"axes", std::vector<int64_t>{4}}})
               ->outputs[0];
    ioup = CreateBaseOp(graph, node, "popart_sigmoid", {ioup}, {})->outputs[0];
    auto *power_0 = CreateConst(graph,
                                node,
                                std::vector<float>{1.0f - iou_aware_factor},
                                {int64_t(1)},
                                VarType2OnnxDType(target_dtype))
                        ->outputs[0];
    auto *power_1 = CreateConst(graph,
                                node,
                                std::vector<float>{iou_aware_factor},
                                {int64_t(1)},
                                VarType2OnnxDType(target_dtype))
                        ->outputs[0];
    ioup = CreateBaseOp(graph, node, "popart_pow", {ioup, power_1}, {})
               ->outputs[0];
    pred_conf =
        CreateBaseOp(graph, node, "popart_pow", {pred_conf, power_0}, {})
            ->outputs[0];
    pred_conf = CreateBaseOp(graph, node, "popart_mul", {pred_conf, ioup}, {})
                    ->outputs[0];
  }
  // pred_conf[pred_conf < conf_thresh] = 0.
  // pred_score = sigmoid(x[:, :, :, :, 5:]) * pred_conf
  // pred_box = pred_box * (pred_conf > 0.).astype('float32')
  auto *value_2 = CreateConst(graph,
                              node,
                              std::vector<float>{2.0f},
                              {int64_t(1)},
                              VarType2OnnxDType(target_dtype))
                      ->outputs[0];
  auto *center =
      CreateBaseOp(graph, node, "popart_div", {pred_box_wh, value_2}, {})
          ->outputs[0];
  auto *min_xy =
      CreateBaseOp(graph, node, "popart_sub", {pred_box_xy, center}, {})
          ->outputs[0];
  auto *max_xy =
      CreateBaseOp(graph, node, "popart_add", {pred_box_xy, center}, {})
          ->outputs[0];

  auto *conf_thresh_node = CreateConst(graph,
                                       node,
                                       std::vector<float>{conf_thresh},
                                       {int64_t(1)},
                                       VarType2OnnxDType(target_dtype))
                               ->outputs[0];
  auto *filter =
      CreateBaseOp(
          graph, node, "popart_greater", {pred_conf, conf_thresh_node}, {})
          ->outputs[0];
  filter = CreateCast(graph, node, {filter}, {}, target_dtype)->outputs[0];
  pred_conf = CreateBaseOp(graph, node, "popart_mul", {pred_conf, filter}, {})
                  ->outputs[0];
  auto *pred_score =
      CreateSlice(graph,
                  node,
                  {transposed_x},
                  {},
                  std::vector<int>{0, 0, 0, 0, 5},
                  std::vector<int>{max_int, max_int, max_int, max_int, max_int},
                  std::vector<int>{0, 1, 2, 3, 4},
                  std::vector<int>{1, 1, 1, 1, 1})
          ->outputs[0];
  pred_score =
      CreateBaseOp(graph, node, "popart_sigmoid", {pred_score}, {})->outputs[0];
  pred_score =
      CreateBaseOp(graph, node, "popart_mul", {pred_score, pred_conf}, {})
          ->outputs[0];
  auto *pred_box = CreateBaseOp(graph,
                                node,
                                "popart_concat",
                                {min_xy, max_xy},
                                {},
                                {{"axis", int64_t(4)}})
                       ->outputs[0];
  pred_box = CreateBaseOp(graph, node, "popart_mul", {pred_box, filter}, {})
                 ->outputs[0];
  pred_box =
      CreateReshape(
          graph, node, {pred_box}, {}, std::vector<int64_t>{nchw[0], -1, 4})
          ->outputs[0];

  // Clip the boxes to img_size
  auto *float_img_size =
      CreateCast(
          graph, node, {GetInputVarNode("ImgSize", node)}, {}, target_dtype)
          ->outputs[0];
  float_img_size = CreateBaseOp(graph,
                                node,
                                "popart_unsqueeze",
                                {float_img_size},
                                {},
                                {{"axes", std::vector<int64_t>(1)}})
                       ->outputs[0];
  auto split_im_hw =
      CreateSplit(
          graph, node, {float_img_size}, {}, std::vector<int64_t>{1, 1}, 2)
          ->outputs;
  auto *im_whwh =
      CreateBaseOp(
          graph,
          node,
          "popart_concat",
          {split_im_hw[1], split_im_hw[0], split_im_hw[1], split_im_hw[0]},
          {},
          {{"axis", int64_t(2)}})
          ->outputs[0];
  if (!clip_bbox) {
    auto *out = CreateBaseOp(graph, node, "popart_mul", {pred_box, im_whwh}, {})
                    ->outputs[0];
    CreateCast(graph,
               node,
               {out},
               {GetOutputVarNode("Boxes", node)},
               GetOutputVarNode("Boxes", node)->Var()->GetDataType());

  } else {
    pred_box = CreateBaseOp(graph, node, "popart_mul", {pred_box, im_whwh}, {})
                   ->outputs[0];
    auto *im_wh = CreateBaseOp(graph,
                               node,
                               "popart_concat",
                               {split_im_hw[1], split_im_hw[0]},
                               {},
                               {{"axis", int64_t(2)}})
                      ->outputs[0];
    auto *float_value_1 = CreateConst(graph,
                                      node,
                                      std::vector<float>{1.0f},
                                      {int64_t(1)},
                                      VarType2OnnxDType(target_dtype))
                              ->outputs[0];
    im_wh = CreateBaseOp(graph, node, "popart_sub", {im_wh, float_value_1}, {})
                ->outputs[0];
    auto pred_box_xymin_xymax =
        CreateSplit(graph, node, {pred_box}, {}, std::vector<int64_t>{2, 2}, 2)
            ->outputs;
    pred_box_xymin_xymax[0] =
        CreateBaseOp(graph, node, "popart_relu", {pred_box_xymin_xymax[0]}, {})
            ->outputs[0];
    pred_box_xymin_xymax[1] =
        CreateBaseOp(
            graph, node, "popart_min", {pred_box_xymin_xymax[1], im_wh}, {})
            ->outputs[0];
    auto *out = CreateBaseOp(graph,
                             node,
                             "popart_concat",
                             pred_box_xymin_xymax,
                             {},
                             {{"axis", int64_t(2)}})
                    ->outputs[0];
    CreateCast(graph,
               node,
               {out},
               {GetOutputVarNode("Boxes", node)},
               GetOutputVarNode("Boxes", node)->Var()->GetDataType());
  }
  auto *score_out = CreateReshape(graph,
                                  node,
                                  {pred_score},
                                  {},
                                  std::vector<int64_t>{nchw[0], -1, class_num})
                        ->outputs[0];
  return CreateCast(graph,
                    node,
                    {score_out},
                    {GetOutputVarNode("Scores", node)},
                    GetOutputVarNode("Scores", node)->Var()->GetDataType());
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(yolo_box, yolo_box_handler);
