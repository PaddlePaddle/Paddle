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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kArgSort = "argsort";
const char *const kArgSortGrad = "argsort_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ArgSortEquivalenceTrans) {
  auto *op = node->Op();
  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  auto descending = PADDLE_GET_CONST(bool, op->GetAttr("descending"));
  auto result =
      builder::ArgSort(*(map_inputs["X"].at(0)), axis, descending, false, true);
  std::vector<std::string> output_names{"Out", "Indices"};
  auto output_name_map = op->Outputs();
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

// Ref:
// Paddle/paddle/fluid/operators/argsort_op_npu.cc
// Paddle/paddle/phi/kernels/cpu/argsort_grad_kernel.cc
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ArgSortGradEquivalenceTrans) {
  builder::Op indices_op = *(map_inputs["Indices"].at(0));
  builder::Op dout_op = *(map_inputs["Out@GRAD"].at(0));
  builder::Op x_op = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  int64_t axis =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  if (axis < 0) axis += x_op.GetType().GetRank();

  auto full_assign = [&](const std::vector<int64_t> &input_dims,
                         const builder::Op input_op,
                         const builder::Op new_indices_op) -> builder::Op {
    const int64_t input_height = std::accumulate(input_dims.begin(),
                                                 input_dims.end() - 1,
                                                 1,
                                                 std::multiplies<int64_t>());
    const int64_t input_width = input_dims[input_dims.size() - 1];

    builder::Op indices_tmp_op =
        builder::Reshape(new_indices_op, {input_height, input_width});

    std::vector<int64_t> indexs_value;
    for (int64_t i = 0; i < input_height; ++i) {
      indexs_value.push_back(i * input_width);
    }
    builder::Op indexs_tmp_op = builder::Const(
        gcu_builder,
        static_cast<void *>(indexs_value.data()),
        builder::Type({input_height, 1},
                      new_indices_op.GetType().GetPrimitiveType()));
    builder::Op indices_index_op = indices_tmp_op + indexs_tmp_op;
    indices_index_op =
        builder::Reshape(indices_index_op, {input_height * input_width, 1});

    builder::Type scalar_type({}, input_op.GetType().GetPrimitiveType());
    std::vector<const char *> region_list{"update_computation"};
    gcu_builder->AddFunc(region_list[0]);
    auto arg0 = gcu_builder->CreateInput(scalar_type, region_list[0]);
    auto arg1 = gcu_builder->CreateInput(scalar_type, region_list[0]);
    auto update = arg1;
    gcu_builder->SetOutput({update}, region_list[0]);

    builder::Op input_tmp_op =
        builder::Reshape(input_op, {input_height * input_width, 1});

    //   builder::ScatterDimensionNumbers scatter_dimension_numbers(
    //       /*update_window_dims=*/{0}, /*inserted_window_dims=*/{},
    //       /*scatter_dims_to_operand_dims=*/{0}, /*index_vector_dim=*/1);
    builder::ScatterDimensionNumbers scatter_dimension_numbers(
        /*update_window_dims=*/{1},
        /*inserted_window_dims=*/{0},
        /*scatter_dims_to_operand_dims=*/{1},
        /*index_vector_dim=*/1);

    return builder::Scatter(input_tmp_op,
                            indices_index_op,
                            input_tmp_op,
                            scatter_dimension_numbers,
                            region_list);
  };

  builder::Op result_op;
  if (axis == x_op.GetType().GetRank() - 1) {
    result_op =
        full_assign(indices_op.GetType().GetShape(), dout_op, indices_op);
  } else {
    std::vector<int64_t> perm(x_op.GetType().GetRank(), 0);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[axis], perm[x_op.GetType().GetRank() - 1]);
    std::vector<int64_t> shapes = x_op.GetType().GetShape();
    std::swap(shapes[axis], shapes[x_op.GetType().GetRank() - 1]);

    auto transpose_dout_op = builder::Transpose(dout_op, perm);
    auto transpose_indices_op = builder::Transpose(indices_op, perm);
    auto transpose_dx_op =
        full_assign(shapes, transpose_dout_op, transpose_indices_op);
    result_op = builder::Transpose(transpose_dx_op, perm);
  }
  result_op = builder::Reshape(result_op, x_op.GetType().GetShape());

  return std::make_shared<GcuOp>(result_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kArgSort, INSENSITIVE, ArgSortEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kArgSortGrad,
                           INSENSITIVE,
                           ArgSortGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
