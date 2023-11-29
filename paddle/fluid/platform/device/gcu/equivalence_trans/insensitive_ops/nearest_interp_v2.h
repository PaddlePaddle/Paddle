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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kNearestInterpV2 = "nearest_interp_v2";
const char *const kNearestInterpV2Grad = "nearest_interp_v2_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               NearestInterpV2EquivalenceTrans) {
  auto *op = node->Op();
  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  if (data_layout != "NCHW" && data_layout != "NHWC") {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "only support NCHW/NHWC for nearest_interp_v2"));
    return nullptr;
  }

  builder::Op input = *(map_inputs["X"].at(0));
  if (data_layout == "NCHW") {
    input = builder::Transpose(input, {0, 2, 3, 1});
  }
  auto input_shape = input.GetType().GetShape();

  int64_t out_h = PADDLE_GET_CONST(int, op->GetAttr("out_h"));
  int64_t out_w = PADDLE_GET_CONST(int, op->GetAttr("out_w"));
  if (map_inputs.count("SizeTensor") != 0) {
    out_h = map_inputs["SizeTensor"].at(0)->GetConstData<int32_t>().at(0);
    out_w = map_inputs["SizeTensor"].at(0)->GetConstData<int32_t>().at(1);
  } else if (map_inputs.count("OutSize") != 0) {
    out_h = map_inputs["OutSize"].at(0)->GetConstData<int32_t>().at(0);
    out_w = map_inputs["OutSize"].at(0)->GetConstData<int32_t>().at(1);
  } else if (map_inputs.count("Scale") != 0) {
    builder::Op scale_op = *(map_inputs["Scale"].at(0));
    if (scale_op.GetType().GetSize() > 1) {
      float scale_h = scale_op.GetConstData<int64_t>()[0];
      float scale_w = scale_op.GetConstData<int64_t>()[1];
      out_h = static_cast<int64_t>(input_shape[1] * scale_h);
      out_w = static_cast<int64_t>(input_shape[2] * scale_w);
    } else {
      float scale = scale_op.GetConstData<int64_t>()[0];
      out_h = static_cast<int64_t>(input_shape[1] * scale);
      out_w = static_cast<int64_t>(input_shape[2] * scale);
    }
  } else {
    auto scale = PADDLE_GET_CONST(std::vector<float>, op->GetAttr("scale"));
    if (scale.size() > 1) {
      out_h = static_cast<int64_t>(input_shape[1] * scale[0]);
      out_w = static_cast<int64_t>(input_shape[2] * scale[1]);
    }
  }

  std::vector<int64_t> output_shape = {
      input_shape[0], out_h, out_w, input_shape[3]};
  builder::Type sizes_type{{4}, builder::PrimitiveType::S64()};
  auto sizes = builder::Const(
      gcu_builder, static_cast<void *>(output_shape.data()), sizes_type);

  auto ptype = input.GetType().GetPrimitiveType();
  builder::Type empty_type{{0}, ptype};
  builder::Op scales = builder::Const(gcu_builder, nullptr, empty_type);

  std::vector<float> roi_val{1};
  builder::Type roi_type(builder::PrimitiveType::S64());
  auto roi = builder::Const(
      gcu_builder, static_cast<void *>(roi_val.data()), roi_type);

  auto align_corners = PADDLE_GET_CONST(bool, op->GetAttr("align_corners"));
  auto align_mode = PADDLE_GET_CONST(int, op->GetAttr("align_mode"));
  auto interp_method =
      PADDLE_GET_CONST(std::string, op->GetAttr("interp_method"));
  if (interp_method != "bilinear") {
    // Based on Comment in `paddle/fluid/operators/interpolate_v2_op.cc:
    // InterpolateV2OpMaker`, align_mode is used for bilinear.
    align_mode = 0;
  }

  std::map<std::string, int64_t> mode_map{
      {"nearest", 0}, {"linear", 1}, {"cubic", 2}};
  std::map<std::string, int64_t> nearest_mode_map{{"simple", 0},
                                                  {"round_prefer_floor", 1},
                                                  {"round_prefer_ceil", 2},
                                                  {"floor", 3},
                                                  {"ceil", 4}};
  std::map<std::string, int64_t> coordinate_transformation_mode_map{
      {"half_pixel", 0},
      {"asymmetric", 1},
      {"pytorch_half_pixel", 2},
      {"tf_half_pixel_for_nn", 3},
      {"align_corners", 4},
      {"tf_crop_and_resize", 5}};
  int64_t mode = mode_map[interp_method];
  int64_t coordinate_transformation_mode = 0;
  if (align_corners) {
    coordinate_transformation_mode = 4;
  } else if (align_mode == 0) {
    coordinate_transformation_mode =
        coordinate_transformation_mode_map["half_pixel"];
  } else if (align_mode == 1) {
    coordinate_transformation_mode =
        coordinate_transformation_mode_map["asymmetric"];
  }

  bool exclude_outside = false;
  int64_t nearest_mode = nearest_mode_map["round_prefer_floor"];
  float extrapolation_value = 0.0;
  float cubic_coeff_a = -0.75;
  std::vector<int64_t> resize_dimensions;
  auto resize = builder::Resize(input,
                                roi,
                                scales,
                                sizes,
                                mode,
                                coordinate_transformation_mode,
                                exclude_outside,
                                nearest_mode,
                                extrapolation_value,
                                cubic_coeff_a,
                                resize_dimensions);

  if (data_layout == "NHWC") {
    return std::make_shared<GcuOp>(resize);
  } else {
    return std::make_shared<GcuOp>(builder::Transpose(resize, {0, 3, 1, 2}));
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               NearestInterpV2GradEquivalenceTrans) {
  auto *op = node->Op();
  auto align_corners = PADDLE_GET_CONST(bool, op->GetAttr("align_corners"));
  auto interp_method =
      PADDLE_GET_CONST(std::string, op->GetAttr("interp_method"));
  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  if (data_layout != "NCHW" && data_layout != "NHWC") {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "only support NCHW/NHWC for nearest_interp_v2_grad"));
  }
  if (align_corners) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "align_corners option can only be set with the interpolating modes: "
        "linear | bilinear | bicubic | trilinear"));
  }
  if (interp_method != "nearest") {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "interp_method error for nearest_interp_v2_grad"));
  }

  builder::Op input = *(map_inputs["X"].at(0));
  builder::Op out_grad = *(map_inputs["Out@GRAD"].at(0));
  if (data_layout == "NCHW") {
    out_grad = builder::Transpose(out_grad, {0, 2, 3, 1});
    input = builder::Transpose(input, {0, 2, 3, 1});
  }

  std::vector<int64_t> in_grad_shape = input.GetType().GetShape();
  builder::Type sizes_type{{4}, builder::PrimitiveType::S64()};
  auto sizes = builder::Const(
      gcu_builder, static_cast<void *>(in_grad_shape.data()), sizes_type);

  auto ptype = input.GetType().GetPrimitiveType();
  builder::Type empty_type{{0}, ptype};
  builder::Op scales = builder::Const(gcu_builder, nullptr, empty_type);

  std::vector<float> roi_val{1};
  builder::Type roi_type(builder::PrimitiveType::S64());
  auto roi = builder::Const(
      gcu_builder, static_cast<void *>(roi_val.data()), roi_type);

  std::map<std::string, int64_t> nearest_mode_map{{"simple", 0},
                                                  {"round_prefer_floor", 1},
                                                  {"round_prefer_ceil", 2},
                                                  {"floor", 3},
                                                  {"ceil", 4}};
  std::map<std::string, int64_t> coordinate_transformation_mode_map{
      {"half_pixel", 0},
      {"asymmetric", 1},
      {"pytorch_half_pixel", 2},
      {"tf_half_pixel_for_nn", 3},
      {"align_corners", 4},
      {"tf_crop_and_resize", 5}};

  int64_t nearest_mode = nearest_mode_map["floor"];
  int64_t coordinate_transformation_mode =
      coordinate_transformation_mode_map["asymmetric"];
  auto resize = builder::ResizeGrad(out_grad,
                                    roi,
                                    scales,
                                    sizes,
                                    /*mode=*/0,
                                    coordinate_transformation_mode,
                                    /*exclude_outside=*/false,
                                    nearest_mode,
                                    /*extrapolation_value=*/0.0,
                                    /*cubic_coeff_a=*/-0.75);

  if (data_layout == "NHWC") {
    return std::make_shared<GcuOp>(resize);
  } else {
    return std::make_shared<GcuOp>(builder::Transpose(resize, {0, 3, 1, 2}));
  }
}

EQUIVALENCE_TRANS_FUNC_REG(kNearestInterpV2,
                           INSENSITIVE,
                           NearestInterpV2EquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kNearestInterpV2Grad,
                           INSENSITIVE,
                           NearestInterpV2GradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
