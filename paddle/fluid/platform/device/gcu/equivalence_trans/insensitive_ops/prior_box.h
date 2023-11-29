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

namespace paddle {
namespace platform {
namespace gcu {
const char *const kPriorBox = "prior_box";

static std::vector<std::vector<float>> ExpandAspectRatios(
    const std::vector<float> &input_aspect_ratior,
    const std::vector<float> &min_sizes,
    const std::vector<float> &max_sizes,
    bool min_max_aspect_ratios_order,
    bool flip) {
  std::vector<float> output_width, output_height;
  if (min_max_aspect_ratios_order) {
    for (size_t s = 0; s < min_sizes.size(); ++s) {
      auto min_size = min_sizes[s];
      output_width.push_back(min_size / 2.0);
      output_height.push_back(min_size / 2.0);
      if (max_sizes.size() > 0) {
        auto max_size = max_sizes[s];
        float sqrt_size = std::sqrt(min_size * max_size) / 2.0;
        output_width.push_back(sqrt_size);
        output_height.push_back(sqrt_size);
      }
      for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
        float ar = input_aspect_ratior[i];
        if (std::fabs(ar - 1.0) < 1e-6) {
          continue;
        }
        output_width.push_back(min_size * std::sqrt(ar) / 2.0);
        output_height.push_back(min_size / std::sqrt(ar) / 2.0);
        if (flip) {
          output_width.push_back(min_size / std::sqrt(ar) / 2.0);
          output_height.push_back(min_size * std::sqrt(ar) / 2.0);
        }
      }
    }
  } else {
    for (size_t s = 0; s < min_sizes.size(); ++s) {
      auto min_size = min_sizes[s];
      output_width.push_back(min_size / 2.0);
      output_height.push_back(min_size / 2.0);
      for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
        float ar = input_aspect_ratior[i];
        if (std::fabs(ar - 1.0) < 1e-6) {
          continue;
        }
        output_width.push_back(min_size * std::sqrt(ar) / 2.0);
        output_height.push_back(min_size / std::sqrt(ar) / 2.0);
        if (flip) {
          output_width.push_back(min_size / std::sqrt(ar) / 2.0);
          output_height.push_back(min_size * std::sqrt(ar) / 2.0);
        }
      }
      if (max_sizes.size() > 0) {
        auto max_size = max_sizes[s];
        float sqrt_size = std::sqrt(min_size * max_size) / 2.0;
        output_width.push_back(sqrt_size);
        output_height.push_back(sqrt_size);
      }
    }
  }
  return std::vector<std::vector<float>>({output_width, output_height});
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, PriorBoxEquivalenceTrans) {
  auto input = *(map_inputs["Input"].at(0));
  auto image = *(map_inputs["Image"].at(0));
  auto op = node->Op();
  auto min_sizes =
      PADDLE_GET_CONST(std::vector<float>, op->GetAttr("min_sizes"));
  auto max_sizes =
      PADDLE_GET_CONST(std::vector<float>, op->GetAttr("max_sizes"));
  auto input_aspect_ratio =
      PADDLE_GET_CONST(std::vector<float>, op->GetAttr("aspect_ratios"));
  auto variances =
      PADDLE_GET_CONST(std::vector<float>, op->GetAttr("variances"));
  auto flip = PADDLE_GET_CONST(bool, op->GetAttr("flip"));
  auto clip = PADDLE_GET_CONST(bool, op->GetAttr("clip"));
  auto min_max_aspect_ratios_order =
      PADDLE_GET_CONST(bool, op->GetAttr("min_max_aspect_ratios_order"));
  auto step_w = PADDLE_GET_CONST(float, op->GetAttr("step_w"));
  auto step_h = PADDLE_GET_CONST(float, op->GetAttr("step_h"));
  auto offset = PADDLE_GET_CONST(float, op->GetAttr("offset"));
  auto input_shape = input.GetType().GetShape();
  auto image_shape = image.GetType().GetShape();
  auto img_width = image_shape[3];
  auto img_height = image_shape[2];
  auto feature_width = input_shape[3];
  auto feature_height = input_shape[2];
  auto dtype = input.GetType().GetPrimitiveType();
  float step_width, step_height;
  if (step_w == 0 || step_h == 0) {
    step_width = static_cast<float>(img_width) / feature_width;
    step_height = static_cast<float>(img_height) / feature_height;
  } else {
    step_width = step_w;
    step_height = step_h;
  }
  auto aspect_boxes = ExpandAspectRatios(input_aspect_ratio,
                                         min_sizes,
                                         max_sizes,
                                         min_max_aspect_ratios_order,
                                         flip);
  auto box_widths = aspect_boxes[0];
  auto box_heights = aspect_boxes[1];
  // 构造（num, H, W）和 (num, W, H) op
  int64_t nums = static_cast<int64_t>(box_heights.size());
  int64_t H = static_cast<int64_t>(feature_height);
  int64_t W = static_cast<int64_t>(feature_width);
  auto hw_op = builder::Iota(gcu_builder, 2, {{nums, H, W}, dtype});
  auto wh_op = builder::Iota(gcu_builder, 2, {{nums, W, H}, dtype});
  auto offset_op = builder::Const(gcu_builder,
                                  static_cast<void *>(&offset),
                                  {{
                                       1,
                                   },
                                   dtype});
  auto step_width_op = builder::Const(gcu_builder,
                                      static_cast<void *>(&step_width),
                                      {{
                                           1,
                                       },
                                       dtype});
  auto step_height_op = builder::Const(gcu_builder,
                                       static_cast<void *>(&step_height),
                                       {{
                                            1,
                                        },
                                        dtype});
  hw_op = (hw_op + offset_op) * step_width_op;
  wh_op = (wh_op + offset_op) * step_height_op;
  // 构造 box op
  auto box_width_op = builder::Const(gcu_builder,
                                     static_cast<void *>(box_widths.data()),
                                     {{nums, 1, 1}, dtype});
  auto box_height_op = builder::Const(gcu_builder,
                                      static_cast<void *>(box_heights.data()),
                                      {{nums, 1, 1}, dtype});
  float img_height_f = static_cast<float>(img_height);
  float img_width_f = static_cast<float>(img_width);
  auto img_width_op = builder::Const(gcu_builder,
                                     static_cast<void *>(&img_width_f),
                                     {{
                                          1,
                                      },
                                      dtype});
  auto img_height_op = builder::Const(gcu_builder,
                                      static_cast<void *>(&img_height_f),
                                      {{
                                           1,
                                       },
                                       dtype});
  auto hw_op_1 = (hw_op - box_width_op) / img_width_op;
  auto hw_op_2 = (hw_op + box_width_op) / img_width_op;
  auto wh_op_1 = (wh_op - box_height_op) / img_height_op;
  auto wh_op_2 = (wh_op + box_height_op) / img_height_op;
  auto t_wh_op_1 = builder::Transpose(wh_op_1, {0, 2, 1});
  auto t_wh_op_2 = builder::Transpose(wh_op_2, {0, 2, 1});
  auto r_hw_op_1 = builder::Reshape(hw_op_1, {{nums, H, W, 1}, dtype});
  auto r_hw_op_2 = builder::Reshape(hw_op_2, {{nums, H, W, 1}, dtype});
  auto rt_wh_op_1 = builder::Reshape(t_wh_op_1, {{nums, H, W, 1}, dtype});
  auto rt_wh_op_2 = builder::Reshape(t_wh_op_2, {{nums, H, W, 1}, dtype});
  auto box_out =
      builder::Concatenate({r_hw_op_1, rt_wh_op_1, r_hw_op_2, rt_wh_op_2}, 3);
  box_out = builder::Transpose(box_out, {1, 2, 0, 3});
  if (clip) {
    auto zero_op = builder::ZerosLike(box_out);
    auto one_op = builder::OnesLike(box_out);
    box_out = builder::Max(box_out, zero_op);
    box_out = builder::Min(box_out, one_op);
  }
  int64_t var_dim = variances.size();
  auto t_var = builder::Const(gcu_builder, variances, {{var_dim}, dtype});
  auto var_out = builder::BroadcastInDim(t_var, {3}, box_out.GetType());

  std::string output_names_attr;
  auto output_name_map = op->Outputs();
  output_names_attr += output_name_map["Boxes"].at(0) + ";";
  output_names_attr += output_name_map["Variances"].at(0);

  auto result_op = builder::Tuple({box_out, var_out});
  result_op.SetAttribute(kAttrOpOutVarName,
                         builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kPriorBox, INSENSITIVE, PriorBoxEquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
