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
const char *const kTranspose = "transpose";
const char *const kTransposeGrad = "transpose_grad";
const char *const kTranspose2 = "transpose2";
const char *const kTranspose2Grad = "transpose2_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TransposeEquivalenceTrans) {
  auto *op = node->Op();
  auto output_name_map = op->Outputs();
  auto out_size = output_name_map.size();
  auto input = *(map_inputs["X"].at(0));
  auto axis = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axis"));
  std::vector<int64_t> perm(axis.begin(), axis.end());
  auto result = builder::Transpose(input, perm);

  if (out_size == 1) {
    return std::make_shared<builder::Op>(result);
  } else {
    auto out_shape_op =
        builder::EmptyLike(input, builder::PrimitiveType::S64());

    std::vector<std::string> output_names{"Out", "XShape"};
    std::string output_names_attr(output_name_map[output_names[0]][0]);
    for (size_t i = 1; i < output_names.size(); ++i) {
      output_names_attr += ";" + output_name_map[output_names[i]][0];
    }
    auto res = builder::Tuple({result, out_shape_op});
    res.SetAttribute(kAttrOpOutVarName,
                     builder::Attribute(output_names_attr.c_str()));

    return std::make_shared<builder::Op>(res);
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               TransposeGradEquivalenceTrans) {
  GcuOp grad_out = *(map_inputs["Out@GRAD"].at(0));
  auto axis = PADDLE_GET_CONST(std::vector<int>, node->Op()->GetAttr("axis"));
  std::vector<int64_t> perm(axis.begin(), axis.end());
  auto out_op = builder::TransposeGrad(grad_out, perm);

  return std::make_shared<GcuOp>(out_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kTranspose, INSENSITIVE, TransposeEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTransposeGrad,
                           INSENSITIVE,
                           TransposeGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTranspose2, INSENSITIVE, TransposeEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTranspose2Grad,
                           INSENSITIVE,
                           TransposeGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
