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

#include <iostream>
#include <memory>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kRoiAlign = "roi_align";
const char *const kRoiAlignGrad = "roi_align_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, RoiAlignEquivalenceTrans) {
  auto input = *(map_inputs["X"].at(0));
  auto rois = *(map_inputs["ROIs"].at(0));
  auto *op = node->Op();
  auto output_height = PADDLE_GET_CONST(int, op->GetAttr("pooled_height"));
  auto output_width = PADDLE_GET_CONST(int, op->GetAttr("pooled_width"));
  auto sampling_ratio = PADDLE_GET_CONST(int, op->GetAttr("sampling_ratio"));
  auto spatial_scale = PADDLE_GET_CONST(float, op->GetAttr("spatial_scale"));
  std::vector<int64_t> dimensions = {0, 1, 2, 3};
  int64_t input_batch = input.GetType().GetShape()[0];
  int64_t rois_batch = rois.GetType().GetShape()[0];

  GcuOp batch_indices;
  std::vector<int64_t> batch_indices_data(rois_batch);
  if (map_inputs.count("RoisNum") != 0 && map_inputs["RoisNum"].size() > 0) {
    auto rois_num = *(map_inputs["RoisNum"].at(0));
    if (rois_num.IsConstant()) {
      std::vector<int32_t> rois_num_data = rois_num.GetConstData<int32_t>();
      int32_t start = 0;
      for (int32_t n = 0; n < input_batch; ++n) {
        for (int32_t i = start; i < start + rois_num_data[n]; ++i) {
          batch_indices_data[i] = n;
        }
        start += rois_num_data[n];
      }
    } else {
      // std::vector<GcuOp> ops;
      // for (int64_t i = 0; i < input_batch; ++i) {
      //   auto slice_op = builder::Slice(rois_num, {i}, {i + 1}, {1});
      //   auto const_op =
      //       builder::Const(gcu_builder,
      //                      i,
      //                      builder::Type({1},
      //                      builder::PrimitiveType::S64()));
      //   auto dyn_brd_op = builder::DynamicBroadcastInDim(const_op, slice_op,
      //   {0}); ops.push_back(dyn_brd_op);
      // }
      // batch_indices = builder::Concatenate(ops, 0);
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported no constant op for roi_align's RoisNum"));
    }
  } else {
    std::vector<int64_t> rois_bod = {0};
    for (int64_t i = 1; i <= input_batch; ++i) {
      rois_bod.emplace_back(rois_bod[rois_bod.size() - 1] + i);
    }
    for (int64_t n = 0; n < input_batch; ++n) {
      for (int64_t i = rois_bod[n]; i < rois_bod[n + 1]; ++i) {
        batch_indices_data[i] = n;
      }
    }
  }

  if (!batch_indices.IsValid())
    batch_indices = builder::Const(
        gcu_builder,
        batch_indices_data,
        builder::Type({rois_batch}, builder::PrimitiveType::S64()));
  auto output = builder::RoiAlign(input,
                                  rois,
                                  batch_indices,
                                  dimensions,
                                  1,
                                  0,
                                  output_height,
                                  output_width,
                                  sampling_ratio > 0 ? sampling_ratio : 0,
                                  spatial_scale);
  return std::make_shared<GcuOp>(output);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, RoiAlignGradEquivalenceTrans) {
  auto grad_out = *(map_inputs["Out@GRAD"].at(0));
  auto input = *(map_inputs["X"].at(0));
  auto rois = *(map_inputs["ROIs"].at(0));
  auto *op = node->Op();
  auto output_height = PADDLE_GET_CONST(int, op->GetAttr("pooled_height"));
  auto output_width = PADDLE_GET_CONST(int, op->GetAttr("pooled_width"));
  auto sampling_ratio = PADDLE_GET_CONST(int, op->GetAttr("sampling_ratio"));
  auto spatial_scale = PADDLE_GET_CONST(float, op->GetAttr("spatial_scale"));
  std::vector<int64_t> dimensions = {0, 1, 2, 3};
  int64_t input_batch = input.GetType().GetShape()[0];
  int64_t rois_batch = rois.GetType().GetShape()[0];

  GcuOp batch_indices;
  std::vector<int64_t> batch_indices_data(rois_batch);
  if (map_inputs.count("RoisNum") != 0 && map_inputs["RoisNum"].size() > 0) {
    auto rois_num = *(map_inputs["RoisNum"].at(0));
    if (rois_num.IsConstant()) {
      std::vector<int32_t> rois_num_data = rois_num.GetConstData<int32_t>();
      int32_t start = 0;
      for (int32_t n = 0; n < input_batch; ++n) {
        for (int32_t i = start; i < start + rois_num_data[n]; ++i) {
          batch_indices_data[i] = n;
        }
        start += rois_num_data[n];
      }
    } else {
      // std::vector<GcuOp> ops;
      // for (int64_t i = 0; i < input_batch; ++i) {
      //   auto slice_op = builder::Slice(rois_num, {i}, {i + 1}, {1});
      //   auto const_op =
      //       builder::Const(gcu_builder,
      //                      i,
      //                      builder::Type({1},
      //                      builder::PrimitiveType::S64()));
      //   auto dyn_brd_op = builder::DynamicBroadcastInDim(const_op, slice_op,
      //   {0}); ops.push_back(dyn_brd_op);
      // }
      // batch_indices = builder::Concatenate(ops, 0);
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported no constant op for roi_align's RoisNum"));
    }
  } else {
    std::vector<int64_t> rois_bod = {0};
    for (int64_t i = 1; i <= input_batch; ++i) {
      rois_bod.emplace_back(rois_bod[rois_bod.size() - 1] + i);
    }
    for (int64_t n = 0; n < input_batch; ++n) {
      for (int64_t i = rois_bod[n]; i < rois_bod[n + 1]; ++i) {
        batch_indices_data[i] = n;
      }
    }
  }

  if (!batch_indices.IsValid())
    batch_indices = builder::Const(
        gcu_builder,
        batch_indices_data,
        builder::Type({rois_batch}, builder::PrimitiveType::S64()));
  auto output = builder::RoiAlignGrad(grad_out,
                                      input,
                                      rois,
                                      batch_indices,
                                      dimensions,
                                      1,
                                      0,
                                      output_height,
                                      output_width,
                                      sampling_ratio > 0 ? sampling_ratio : 0,
                                      spatial_scale);
  return std::make_shared<GcuOp>(output);
}

EQUIVALENCE_TRANS_FUNC_REG(kRoiAlign, INSENSITIVE, RoiAlignEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kRoiAlignGrad,
                           INSENSITIVE,
                           RoiAlignGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
