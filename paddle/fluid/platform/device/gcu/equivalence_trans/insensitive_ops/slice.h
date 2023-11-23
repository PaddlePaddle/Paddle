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

#include <algorithm>
#include <memory>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kSlice = "slice";
const char *const kSliceGrad = "slice_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SliceEquivalenceTrans) {
  builder::Op input = *(map_inputs["Input"].at(0));
  auto *op = node->Op();
  auto axes = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  auto starts_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("starts"));
  auto ends_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("ends"));
  auto decrease_axis_i32 =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("decrease_axis"));
  std::vector<int64_t> decrease_axis;
  for (auto dim : decrease_axis_i32) {
    decrease_axis.emplace_back(static_cast<int64_t>(dim));
  }

  if (map_inputs.count("StartsTensor") != 0) {
    starts_i32 = map_inputs["StartsTensor"].at(0)->GetConstData<int>();
  } else if (map_inputs.count("StartsTensorList") != 0) {
    std::vector<int> dims;
    for (size_t i = 0; i < map_inputs["StartsTensorList"].size(); ++i) {
      dims.emplace_back(
          map_inputs["StartsTensorList"].at(0)->GetConstData<int64_t>()[0]);
    }
    starts_i32 = dims;
  }
  if (map_inputs.count("EndsTensor") != 0) {
    ends_i32 = map_inputs["EndsTensor"].at(0)->GetConstData<int>();
  } else if (map_inputs.count("EndsTensorList") != 0) {
    std::vector<int> dims;
    for (size_t i = 0; i < map_inputs["EndsTensorList"].size(); ++i) {
      dims.emplace_back(
          map_inputs["EndsTensorList"].at(0)->GetConstData<int64_t>()[0]);
    }
    ends_i32 = dims;
  }

  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  for (auto dim : starts_i32) {
    starts.emplace_back(static_cast<int64_t>(dim));
  }
  for (auto dim : ends_i32) {
    ends.emplace_back(static_cast<int64_t>(dim));
  }
  auto rank = input.GetType().GetRank();
  const std::vector<int64_t> &input_shapes = input.GetType().GetShape();
  std::vector<int64_t> start_indices(rank, 0);
  std::vector<int64_t> limit_indices = input_shapes;
  for (size_t i = 0; i < axes.size(); ++i) {
    int dim = axes[i];
    if (dim < 0) {
      dim += rank;
    }
    start_indices[dim] =
        starts[i] < 0 ? starts[i] + input_shapes[dim] : starts[i];
    start_indices[dim] = std::max(start_indices[dim], 0L);
    start_indices[dim] = std::min(start_indices[dim], input_shapes[dim]);

    limit_indices[dim] = ends[i] < 0 ? ends[i] + input_shapes[dim] : ends[i];
    limit_indices[dim] = std::min(limit_indices[dim], input_shapes[dim]);
    limit_indices[dim] = std::max(limit_indices[dim], 0L);
  }
  std::vector<int64_t> strides(rank, 1);
  // for transpose erase:
  std::vector<int64_t> perm_cfirst_to_clast;
  std::vector<int64_t> perm_clast_to_cfirst;
  std::vector<int64_t> transed_start_indices = start_indices;
  std::vector<int64_t> transed_limit_indices = limit_indices;
  std::vector<int64_t> transed_strides = strides;
  std::vector<int64_t> transed_decrease_axis = decrease_axis;

  auto GetShapeByPerm = [](const std::vector<int64_t> &shape,
                           const std::vector<int64_t> &perm) {
    std::vector<int64_t> ret;
    PADDLE_ENFORCE_EQ(
        shape.size(),
        perm.size(),
        platform::errors::Fatal("When Slice op do trans shape, in shape "
                                "size[%zu] shoule be same with perm size[%zu]",
                                shape.size(),
                                perm.size()));
    for (const auto &idx : perm) {
      ret.push_back(shape[idx]);
    }
    return ret;
  };
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      perm_cfirst_to_clast = {0, 2, 3, 1};
      perm_clast_to_cfirst = {0, 3, 1, 2};
      input = builder::Transpose(input, perm_cfirst_to_clast);
      transed_start_indices =
          GetShapeByPerm(start_indices, perm_cfirst_to_clast);
      transed_limit_indices =
          GetShapeByPerm(limit_indices, perm_cfirst_to_clast);
      transed_strides = GetShapeByPerm(strides, perm_cfirst_to_clast);
    } else if (rank == 5) {
      perm_cfirst_to_clast = {0, 2, 3, 4, 1};
      perm_clast_to_cfirst = {0, 4, 1, 2, 3};
      input = builder::Transpose(input, perm_cfirst_to_clast);
      transed_start_indices =
          GetShapeByPerm(start_indices, perm_cfirst_to_clast);
      transed_limit_indices =
          GetShapeByPerm(limit_indices, perm_cfirst_to_clast);
      transed_strides = GetShapeByPerm(strides, perm_cfirst_to_clast);
    }
  }

  auto slice = builder::Slice(
      input, transed_start_indices, transed_limit_indices, transed_strides);
  if ((rank == 4 || rank == 5) && (running_mode == RunningMode::ADAPTIVE)) {
    slice = builder::Transpose(slice, perm_clast_to_cfirst);
  }
  if (decrease_axis.size() == 0) {
    return std::make_shared<GcuOp>(slice);
  } else {
    auto slice_shape = slice.GetType().GetShape();
    std::vector<int64_t> new_shape;
    size_t iter = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(slice_shape.size()); ++i) {
      if (iter < decrease_axis.size() && i == decrease_axis[iter]) {
        ++iter;
      } else {
        new_shape.emplace_back(slice_shape[i]);
      }
    }
    if (new_shape.empty()) {
      new_shape.emplace_back(1);
    }
    return std::make_shared<GcuOp>(builder::Reshape(slice, new_shape));
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SliceGradEquivalenceTrans) {
  builder::Op input_op = *(map_inputs["Input"].at(0));
  builder::Op dout_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  auto axes = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  auto starts = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("starts"));
  auto ends = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("ends"));
  auto decrease_axis =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("decrease_axis"));

  if (map_inputs.count("StartsTensor") != 0 &&
      map_inputs["StartsTensor"].size() > 0) {
    starts = map_inputs["StartsTensor"].at(0)->GetConstData<int>();
  } else if (map_inputs.count("StartsTensorList") != 0 &&
             map_inputs["StartsTensorList"].size() > 0) {
    std::vector<int> dims;
    for (size_t i = 0; i < map_inputs["StartsTensorList"].size(); ++i) {
      dims.emplace_back(
          map_inputs["StartsTensorList"].at(0)->GetConstData<int64_t>()[0]);
    }
    starts = dims;
  }
  if (map_inputs.count("EndsTensor") != 0 &&
      map_inputs["EndsTensor"].size() > 0) {
    ends = map_inputs["EndsTensor"].at(0)->GetConstData<int>();
  } else if (map_inputs.count("EndsTensorList") != 0 &&
             map_inputs["EndsTensorList"].size() > 0) {
    std::vector<int> dims;
    for (size_t i = 0; i < map_inputs["EndsTensorList"].size(); ++i) {
      dims.emplace_back(
          map_inputs["EndsTensorList"].at(0)->GetConstData<int64_t>()[0]);
    }
    ends = dims;
  }

  const int64_t input_rank = input_op.GetType().GetRank();
  const std::vector<int64_t> &input_shapes = input_op.GetType().GetShape();

  std::vector<int64_t> padding_low(input_rank, 0);
  std::vector<int64_t> padding_high(input_rank, 0);
  std::vector<int64_t> padding_interior(input_rank, 0);
  int64_t cnt = 0;
  for (int64_t i = 0; i < input_rank; ++i) {
    int64_t axis = cnt < static_cast<int64_t>(axes.size()) ? axes[cnt] : -1;
    if (axis == i) {
      int64_t start =
          starts[cnt] < 0 ? starts[cnt] + input_shapes[i] : starts[cnt];
      start = std::max(start, 0L);
      start = std::min(start, input_shapes[i]);
      int64_t end = ends[cnt] < 0 ? ends[cnt] + input_shapes[i] : ends[cnt];
      end = std::min(end, input_shapes[i]);
      end = std::max(end, 0L);

      padding_low[i] = start;
      padding_high[i] = input_shapes[i] - end;
      ++cnt;
    }
  }

  builder::Op padding_value_op;
  {
    auto input_dtype = input_op.GetType().GetPrimitiveType();
    if (input_dtype == builder::PrimitiveType::F32()) {
      float const_0_data = 0.0f;
      padding_value_op =
          builder::Const(gcu_builder,
                         static_cast<void *>(&const_0_data),
                         builder::Type(builder::PrimitiveType::F32()));
    } else if (input_dtype == builder::PrimitiveType::F64()) {
      double const_0_data = 0.0;
      padding_value_op =
          builder::Const(gcu_builder,
                         static_cast<void *>(&const_0_data),
                         builder::Type(builder::PrimitiveType::F64()));
    } else if (input_dtype == builder::PrimitiveType::S32()) {
      int32_t const_0_data = 0;
      padding_value_op =
          builder::Const(gcu_builder,
                         static_cast<void *>(&const_0_data),
                         builder::Type(builder::PrimitiveType::S32()));
    } else if (input_dtype == builder::PrimitiveType::S64()) {
      int64_t const_0_data = 0;
      padding_value_op =
          builder::Const(gcu_builder,
                         static_cast<void *>(&const_0_data),
                         builder::Type(builder::PrimitiveType::S64()));
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "GCU slice_grad NOT support data type."));
    }
  }

  if (decrease_axis.size() != 0) {
    std::sort(decrease_axis.begin(), decrease_axis.end());
    auto dout_shape = dout_op.GetType().GetShape();
    std::vector<int64_t> new_shape;
    size_t iter = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(dout_shape.size()); ++i) {
      if (iter < decrease_axis.size() && i == decrease_axis[iter]) {
        new_shape.emplace_back(1);
        ++iter;
      }
      new_shape.emplace_back(dout_shape[i]);
    }
    dout_op = builder::Reshape(dout_op, new_shape);
  }

  // for transpose erase:
  std::vector<int64_t> perm_cfirst_to_clast;
  std::vector<int64_t> perm_clast_to_cfirst;
  std::vector<int64_t> transed_padding_low = padding_low;
  std::vector<int64_t> transed_padding_high = padding_high;
  std::vector<int64_t> transed_padding_interior = padding_interior;

  auto GetShapeByPerm = [](const std::vector<int64_t> &shape,
                           const std::vector<int64_t> &perm) {
    std::vector<int64_t> ret;
    PADDLE_ENFORCE_EQ(
        shape.size(),
        perm.size(),
        platform::errors::Fatal("When Slice op do trans shape, in shape "
                                "size[%zu] shoule be same with perm size[%zu]",
                                shape.size(),
                                perm.size()));
    for (const auto &idx : perm) {
      ret.push_back(shape[idx]);
    }
    return ret;
  };
  if (running_mode == RunningMode::ADAPTIVE) {
    if (input_rank == 4) {
      perm_cfirst_to_clast = {0, 2, 3, 1};
      perm_clast_to_cfirst = {0, 3, 1, 2};
      dout_op = builder::Transpose(dout_op, perm_cfirst_to_clast);
      transed_padding_low = GetShapeByPerm(padding_low, perm_cfirst_to_clast);
      transed_padding_high = GetShapeByPerm(padding_high, perm_cfirst_to_clast);
      transed_padding_interior =
          GetShapeByPerm(padding_interior, perm_cfirst_to_clast);
      auto pad_op = builder::Pad(dout_op,
                                 padding_value_op,
                                 0 /*constant mode*/,
                                 transed_padding_low,
                                 transed_padding_high,
                                 transed_padding_interior);
      return std::make_shared<GcuOp>(
          builder::Transpose(pad_op, perm_clast_to_cfirst));
    } else if (input_rank == 5) {
      perm_cfirst_to_clast = {0, 2, 3, 4, 1};
      perm_clast_to_cfirst = {0, 4, 1, 2, 3};
      dout_op = builder::Transpose(dout_op, perm_cfirst_to_clast);
      transed_padding_low = GetShapeByPerm(padding_low, perm_cfirst_to_clast);
      transed_padding_high = GetShapeByPerm(padding_high, perm_cfirst_to_clast);
      transed_padding_interior =
          GetShapeByPerm(padding_interior, perm_cfirst_to_clast);
      auto pad_op = builder::Pad(dout_op,
                                 padding_value_op,
                                 0 /*constant mode*/,
                                 transed_padding_low,
                                 transed_padding_high,
                                 transed_padding_interior);
      return std::make_shared<GcuOp>(
          builder::Transpose(pad_op, perm_clast_to_cfirst));
    }
  }
  return std::make_shared<GcuOp>(builder::Pad(dout_op,
                                              padding_value_op,
                                              0 /*constant mode*/,
                                              padding_low,
                                              padding_high,
                                              padding_interior));
}

EQUIVALENCE_TRANS_FUNC_REG(kSlice, INSENSITIVE, SliceEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSliceGrad, INSENSITIVE, SliceGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
