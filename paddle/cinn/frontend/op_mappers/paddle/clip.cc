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

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ClipOpMapper(const paddle::cpp::OpDesc& op_desc,
                  const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of clip op must be 1."));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Out").size(),
      1UL,
      phi::errors::InvalidArgument("The output of clip op must be 1."));
  auto out_name = op_desc.Output("Out").front();
  auto x = ctx.GetVar(x_name);
  auto builder = ctx.Builder();

  if (op_desc.HasInput("Min") && op_desc.Input("Min").size() > 0) {
    PADDLE_ENFORCE_EQ(op_desc.Input("Min").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "clip op should have only one input for Min"));
    auto min_val_name = op_desc.Input("Min").front();
    auto min_val_tensor = ctx.GetVar(min_val_name);
    PADDLE_ENFORCE_EQ(min_val_tensor->shape == cinn::utils::ShapeType{1},
                      true,
                      phi::errors::InvalidArgument(
                          "The [Min] tensor shape of clip op should be [1]."));
    if (x->type != min_val_tensor->type) {
      min_val_tensor =
          builder->Cast(min_val_tensor, cinn::common::Type2Str(x->type));
    }
    min_val_tensor = builder->BroadcastTo(min_val_tensor, x->shape);
    x = builder->Max(x, min_val_tensor);
  } else {
    PADDLE_ENFORCE_EQ(
        op_desc.HasAttr("min"),
        true,
        phi::errors::InvalidArgument(
            "The clip op should has [min] attribute or [Min] tensor input."));
    auto min_value = op_desc.GetAttr<float>("min");
    auto min_val_tensor =
        builder->FillConstant(x->shape,
                              min_value,
                              cinn::common::UniqName(x->id + "_min"),
                              cinn::common::Type2Str(x->type));
    x = builder->Max(x, min_val_tensor);
  }

  if (op_desc.HasInput("Max") && op_desc.Input("Max").size() > 0) {
    PADDLE_ENFORCE_EQ(op_desc.Input("Max").size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "clip op should have only one input for Max"));
    auto max_val_name = op_desc.Input("Max").front();
    auto max_val_tensor = ctx.GetVar(max_val_name);
    PADDLE_ENFORCE_EQ(max_val_tensor->shape == cinn::utils::ShapeType{1},
                      true,
                      phi::errors::InvalidArgument(
                          "The [Max] tensor shape of clip op should be [1]."));
    if (x->type != max_val_tensor->type) {
      max_val_tensor =
          builder->Cast(max_val_tensor, cinn::common::Type2Str(x->type));
    }
    max_val_tensor = builder->BroadcastTo(max_val_tensor, x->shape);
    x = builder->Min(x, max_val_tensor);
  } else {
    PADDLE_ENFORCE_EQ(
        op_desc.HasAttr("max"),
        true,
        phi::errors::InvalidArgument(
            "The clip op should has [max] attribute or [Max] tensor input."));
    auto max_value = op_desc.GetAttr<float>("max");
    auto max_val_tensor =
        builder->FillConstant(x->shape,
                              max_value,
                              cinn::common::UniqName("constant"),
                              cinn::common::Type2Str(x->type));
    x = builder->Min(x, max_val_tensor);
  }

  ctx.AddVar(out_name, x);
  ctx.AddVarModelToProgram(out_name, x->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_clip) {
  CINN_REGISTER_OP_MAPPER(clip, cinn::frontend::paddle_mappers::ClipOpMapper)
  return true;
}
