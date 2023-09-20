// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <variant>

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ScaleOpMapper(const paddle::cpp::OpDesc& op_desc,
                   const cinn::frontend::OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto bias = utils::GetAttrOrDefault<float>(op_desc, "bias", 0.0f);
  auto bias_after_scale =
      utils::GetAttrOrDefault<bool>(op_desc, "bias_after_scale", true);

  auto x = ctx.GetVar(x_name);

  absl::optional<Variable> out;
  if (op_desc.HasInput("ScaleTensor") &&
      !op_desc.Input("ScaleTensor").empty()) {
    CHECK_EQ(op_desc.Input("ScaleTensor").size(), 1);
    auto scale_name = op_desc.Input("ScaleTensor").front();
    auto scale_tensor = ctx.GetVar(scale_name);

    VLOG(4) << out_name << " = scale(" << x_name << "=" << x
            << ", scale=" << scale_name << "[" << scale_tensor
            << "], bias=" << bias << ", bias_after_scale=" << bias_after_scale;

    CHECK(scale_tensor->shape == cinn::utils::ShapeType{1})
        << "The shape of [ScaleTensor] should be [1], but here ["
        << cinn::utils::Join(scale_tensor->shape, ", ") << "]";
    scale_tensor = ctx.Builder()->Cast(scale_tensor, common::Type2Str(x->type));
    scale_tensor = ctx.Builder()->BroadcastTo(scale_tensor, x->shape);

    if (bias != 0.0f) {
      auto bias_tensor = ctx.Builder()->FillConstant(
          x->shape, bias, x->id + "_bias", common::Type2Str(x->type));
      if (bias_after_scale) {
        out = ctx.Builder()->Add(bias_tensor,
                                 ctx.Builder()->Multiply(x, scale_tensor));
      } else {
        out = ctx.Builder()->Multiply(scale_tensor,
                                      ctx.Builder()->Add(x, bias_tensor));
      }
    } else {
      out = ctx.Builder()->Multiply(scale_tensor, x);
    }
  } else {
    auto scale = utils::GetAttrOrDefault<float>(op_desc, "scale", 1.0f);

    VLOG(4) << out_name << " = scale(" << x_name << "=" << x
            << ", scale=" << scale << ", bias=" << bias
            << ", bias_after_scale=" << bias_after_scale;

    out = ctx.Builder()->Scale(x, scale, bias, bias_after_scale);
  }

  ctx.AddVar(out_name, out.value());
  ctx.AddVarModelToProgram(out_name, out.value()->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_scale) {
  CINN_REGISTER_OP_MAPPER(scale, cinn::frontend::paddle_mappers::ScaleOpMapper)
  return true;
}
