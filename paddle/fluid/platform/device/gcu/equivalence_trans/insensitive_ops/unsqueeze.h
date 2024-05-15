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
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kunsqueeze = "unsqueeze";
const char *const kunsqueezeGrad = "unsqueeze_grad";
const char *const kunsqueeze2 = "unsqueeze2";
const char *const kunsqueeze2Grad = "unsqueeze2_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, unsqueezeEquivalenceTrans) {
  auto *op = node->Op();
  GcuOp input = *(map_inputs["X"].at(0));
  auto axes = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axes"));

  GcuOp axes_op;
  if (map_inputs.count("AxesTensor") != 0) {
    axes_op = *(map_inputs["AxesTensor"].at(0));
  } else if (map_inputs.count("AxesTensorList") != 0) {
    std::vector<GcuOp> axes_list;
    for (size_t i = 0; i < map_inputs["AxesTensorList"].size(); ++i) {
      auto dim_op = *(map_inputs["AxesTensorList"].at(i));
      const int64_t rank = dim_op.GetType().GetRank();
      PADDLE_ENFORCE_LE(
          rank,
          1,
          platform::errors::InvalidArgument(
              "unsqueeze AxesTensor's rank must <= 1, but got: %d", rank));
      if (rank == 0) dim_op = builder::Reshape(dim_op, {1});
      axes_list.emplace_back(dim_op);
    }
    axes_op = builder::Concatenate(axes_list, 0);
  } else if (!axes.empty()) {
    axes_op = builder::Const(input.GetBuilder(),
                             static_cast<void *>(axes.data()),
                             builder::Type({static_cast<int64_t>(axes.size())},
                                           builder::PrimitiveType::S32()));
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("Unsupported unsqueeze without axes"));
  }

  auto result = builder::Unsqueeze(input, axes_op);
  auto output_name_map = op->Outputs();
  if (output_name_map.size() == 1) {
    return std::make_shared<builder::Op>(result);
  } else {
    auto input_shape_op =
        builder::EmptyLike(input, builder::PrimitiveType::S64());
    auto res = builder::Tuple({result, input_shape_op});

    std::vector<std::string> output_names{"Out", "XShape"};
    std::string output_names_attr(output_name_map[output_names[0]][0]);
    for (size_t i = 1; i < output_names.size(); ++i) {
      output_names_attr += ";" + output_name_map[output_names[i]][0];
    }
    res.SetAttribute(kAttrOpOutVarName,
                     builder::Attribute(output_names_attr.c_str()));

    return std::make_shared<GcuOp>(res);
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               unsqueezeGradEquivalenceTrans) {
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  GcuOp input = *(map_inputs["X"].at(0));
  auto out_op = builder::Reshape(out_grad, input.GetType());
  return std::make_shared<GcuOp>(out_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               unsqueeze2GradEquivalenceTrans) {
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  GcuOp in_shape = *(map_inputs["XShape"].at(0));

  std::vector<int64_t> src_shape = in_shape.GetType().GetShape();
  std::vector<int64_t> tmp_shape(src_shape.begin() + 1, src_shape.end());
  builder::Type output_type(tmp_shape, out_grad.GetType().GetPrimitiveType());
  auto out_op = builder::Reshape(out_grad, output_type);

  return std::make_shared<GcuOp>(out_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kunsqueeze, INSENSITIVE, unsqueezeEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kunsqueezeGrad,
                           INSENSITIVE,
                           unsqueezeGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kunsqueeze2, INSENSITIVE, unsqueezeEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kunsqueeze2Grad,
                           INSENSITIVE,
                           unsqueeze2GradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
