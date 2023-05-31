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

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ClipOpMapper(const paddle::cpp::OpDesc& op_desc,
                  const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  auto x = ctx.GetVar(x_name);
  auto builder = ctx.Builder();

  if (op_desc.HasInput("Min") && op_desc.Input("Min").size() > 0) {
    CHECK_EQ(op_desc.Input("Min").size(), 1)
        << "clip op should have only one input for Min";
    auto min_val_name = op_desc.Input("Min").front();
    auto min_val_tensor = ctx.GetVar(min_val_name);
    CHECK(min_val_tensor->shape == cinn::utils::ShapeType{1})
        << "The [Min] tensor shape of clip op should be [1], but here ["
        << cinn::utils::Join(min_val_tensor->shape, ", ") << "]";
    if (x->type != min_val_tensor->type) {
      min_val_tensor = builder->Cast(min_val_tensor, common::Type2Str(x->type));
    }
    min_val_tensor = builder->BroadcastTo(min_val_tensor, x->shape);
    x = builder->Max(x, min_val_tensor);
  } else {
    CHECK(op_desc.HasAttr("min"))
        << "The clip op should has [min] attribute or [Min] tensor input.";
    auto min_value = op_desc.GetAttr<float>("min");
    auto min_val_tensor =
        builder->FillConstant(x->shape,
                              min_value,
                              common::UniqName(x->id + "_min"),
                              common::Type2Str(x->type));
    x = builder->Max(x, min_val_tensor);
  }

  if (op_desc.HasInput("Max") && op_desc.Input("Max").size() > 0) {
    CHECK_EQ(op_desc.Input("Max").size(), 1)
        << "clip op should have only one input for Max";
    auto max_val_name = op_desc.Input("Max").front();
    auto max_val_tensor = ctx.GetVar(max_val_name);
    CHECK(max_val_tensor->shape == cinn::utils::ShapeType{1})
        << "The [Max] tensor shape of clip op should be [1], but here ["
        << cinn::utils::Join(max_val_tensor->shape, ", ") << "]";
    if (x->type != max_val_tensor->type) {
      max_val_tensor = builder->Cast(max_val_tensor, common::Type2Str(x->type));
    }
    max_val_tensor = builder->BroadcastTo(max_val_tensor, x->shape);
    x = builder->Min(x, max_val_tensor);
  } else {
    CHECK(op_desc.HasAttr("max"))
        << "The clip op should has [max] attribute or [Max] tensor input.";
    auto max_value = op_desc.GetAttr<float>("max");
    auto max_val_tensor = builder->FillConstant(x->shape,
                                                max_value,
                                                common::UniqName("constant"),
                                                common::Type2Str(x->type));
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
