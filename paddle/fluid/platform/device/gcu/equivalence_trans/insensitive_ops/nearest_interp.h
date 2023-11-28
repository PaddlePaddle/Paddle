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
const char *const kNearestInterp = "nearest_interp";
const char *const kNearestInterpGrad = "nearest_interp_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               NearestInterpEquivalenceTrans) {
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

  builder::Op sizes;
  if (map_inputs.count("SizeTensor") != 0) {
    std::vector<int64_t> dims;
    for (size_t i = 0; i < map_inputs["SizeTensor"].size(); ++i) {
      auto dim = map_inputs["SizeTensor"].at(0)->GetConstData<int32_t>()[0];
      dims.emplace_back(static_cast<int64_t>(dim));
    }
    std::vector<int64_t> output_shape = {
        input_shape[0], dims[0], dims[1], input_shape[3]};
    builder::Type sizes_type{{4}, builder::PrimitiveType::S64()};
    sizes = builder::Const(
        gcu_builder, static_cast<void *>(output_shape.data()), sizes_type);
  }
  if (!sizes.IsValid() && map_inputs.count("OutSize") != 0) {
    auto dims = map_inputs["OutSize"].at(0)->GetConstData<int32_t>();
    std::vector<int64_t> output_shape = {input_shape[0],
                                         static_cast<int64_t>(dims[0]),
                                         static_cast<int64_t>(dims[1]),
                                         input_shape[3]};
    builder::Type sizes_type{{4}, builder::PrimitiveType::S64()};
    sizes = builder::Const(
        gcu_builder, static_cast<void *>(output_shape.data()), sizes_type);
  }
  if (!sizes.IsValid()) {
    auto out_h = PADDLE_GET_CONST(int, op->GetAttr("out_h"));
    auto out_w = PADDLE_GET_CONST(int, op->GetAttr("out_w"));
    if (out_h > 0 && out_w > 0) {
      std::vector<int64_t> output_shape = {
          input_shape[0], out_h, out_w, input_shape[3]};
      builder::Type sizes_type{{4}, builder::PrimitiveType::S64()};
      sizes = builder::Const(
          gcu_builder, static_cast<void *>(output_shape.data()), sizes_type);
    }
  }
  if (!sizes.IsValid()) {
    float scale = 0.0;
    if (map_inputs.count("Scale") != 0) {
      scale = map_inputs["Scale"].at(0)->GetConstData<float>()[0];
    } else {
      scale = PADDLE_GET_CONST(float, op->GetAttr("scale"));
    }
    int64_t out_h =
        static_cast<int64_t>(static_cast<float>(input_shape[1]) * scale);
    int64_t out_w =
        static_cast<int64_t>(static_cast<float>(input_shape[2]) * scale);
    std::vector<int64_t> output_shape = {
        input_shape[0], out_h, out_w, input_shape[3]};
    builder::Type sizes_type{{4}, builder::PrimitiveType::S64()};
    sizes = builder::Const(
        gcu_builder, static_cast<void *>(output_shape.data()), sizes_type);
  }
  auto ptype = input.GetType().GetPrimitiveType();
  builder::Type empty_type{{0}, ptype};
  builder::Op scales = builder::Const(gcu_builder, nullptr, empty_type);

  std::vector<int64_t> roi_val{1};
  builder::Type roi_type(builder::PrimitiveType::S64());
  auto roi = builder::Const(
      gcu_builder, static_cast<void *>(roi_val.data()), roi_type);

  auto align_corners = PADDLE_GET_CONST(bool, op->GetAttr("align_corners"));
  auto align_mode = PADDLE_GET_CONST(int, op->GetAttr("align_mode"));
  auto interp_method =
      PADDLE_GET_CONST(std::string, op->GetAttr("interp_method"));

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
  int64_t nearest_mode = nearest_mode_map["round_prefer_ceil"];
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
                               NearestInterpGradEquivalenceTrans) {
  auto *op = node->Op();
  auto align_corners = PADDLE_GET_CONST(bool, op->GetAttr("align_corners"));
  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  if (data_layout != "NCHW" && data_layout != "NHWC") {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "only support NCHW/NHWC for nearest_interp_grad"));
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
  if (align_corners) {
    nearest_mode = nearest_mode_map["round_prefer_floor"];
    coordinate_transformation_mode =
        coordinate_transformation_mode_map["align_corners"];
  }
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

EQUIVALENCE_TRANS_FUNC_REG(kNearestInterp,
                           INSENSITIVE,
                           NearestInterpEquivalenceTrans);

EQUIVALENCE_TRANS_FUNC_REG(kNearestInterpGrad,
                           INSENSITIVE,
                           NearestInterpGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
