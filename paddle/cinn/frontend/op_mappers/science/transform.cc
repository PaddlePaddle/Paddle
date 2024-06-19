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

#include <absl/types/optional.h>

#include <functional>
#include <numeric>

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/cinn/frontend/var_type_utils.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace science_mappers {

using cinn::utils::ShapeType;

void ConcatOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  PADDLE_ENFORCE_GE(op_desc.Input("XS").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "The input of concat op must be at least 1"));
  auto x_names = op_desc.Input("XS");
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The output of concat op must be 1"));
  auto out_name = op_desc.Output("Y").front();

  Variable out;
  if (x_names.size() == 1) {
    // if concat only has one input, using Identity to copy the input and return
    auto x = ctx.GetVar(x_names.front());
    out = ctx.Builder()->Identity(x);
  } else {
    std::vector<Variable> xs;
    for (const auto& name : x_names) {
      xs.emplace_back(ctx.GetVar(name));
    }

    auto axis =
        utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

    out = ctx.Builder()->Concat(xs, axis);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SplitOpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of split op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_GE(op_desc.Output("YS").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "The output of split op must be at least 1"));
  auto out_name = op_desc.Output("YS");

  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("num_or_sections"),
      true,
      phi::errors::InvalidArgument(
          "The split_p operator should has 'num_or_sections' attribute."));
  auto num_or_sections =
      utils::ToShapeType(utils::GetAttrOrDefault<std::vector<int64_t>>(
          op_desc, "num_or_sections"));

  PADDLE_ENFORCE_EQ(!num_or_sections.empty(),
                    true,
                    phi::errors::InvalidArgument(
                        "The Split op cannot found [num_or_sections] "
                        "attribute!  ! Please check."));

  auto axis =
      utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

  auto x = ctx.GetVar(x_name);

  auto x_shape = x->shape;
  if (num_or_sections.size() == 1U) {
    PADDLE_ENFORCE_EQ(
        x_shape[axis] % num_or_sections[0],
        0,
        phi::errors::InvalidArgument(
            "If the attribute 'num_or_sections' is a number, it should be "
            "divisible by the "
            "axis's dimension of inputs A ! Please check."));
  } else {
    cinn::utils::DimType sec_sum = 0;
    bool has_neg = false;
    for (auto sec : num_or_sections) {
      if (sec > 0) {
        sec_sum += sec;
      } else if (sec == -1 && !has_neg) {
        has_neg = true;
      } else if (sec == 0) {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "The attribute 'num_or_sections' of split should not has "
            "0 ! Please check."));
      } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "The attribute 'num_or_sections' of split can only have "
            "at most one '-1' ! Please check."));
      }
    }
    PADDLE_ENFORCE_EQ(!has_neg && sec_sum == x_shape[axis],
                      true,
                      phi::errors::InvalidArgument(
                          "The sum of attr sections should be equal with the "
                          "axis's dimension "
                          "value of inputs A in Split ! Please check."));
  }

  VLOG(4) << "Split " << x_name << " with shape ("
          << cinn::utils::Join(x->shape, ",") << ") "
          << " to section (" << cinn::utils::Join(num_or_sections, ",")
          << ") at dimension " << axis;

  auto out = ctx.Builder()->Split(x, num_or_sections, axis);

  PADDLE_ENFORCE_EQ(out.size(),
                    out_name.size(),
                    phi::errors::InvalidArgument(
                        "The Split op should has %d output, but only %d",
                        out_name.size(),
                        out.size()));

  for (int i = 0; i < out.size(); ++i) {
    ctx.AddVar(out_name[i], out[i]);
    ctx.AddVarModelToProgram(out_name[i], out[i]->id);
  }
}

void ReshapeOpMapper(const paddle::cpp::OpDesc& op_desc,
                     const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of reshape op must be 1"));
  auto x_name = op_desc.Input("X").front();

  auto shape = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "shape"));

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Reshape " << x_name << "from shape ("
          << cinn::utils::Join(x->shape, ",") << ") to ("
          << cinn::utils::Join(shape, ",") << ").";

  auto out = ctx.Builder()->Reshape(x, shape);

  PADDLE_ENFORCE_EQ(
      op_desc.Output("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The output of reshape op must be 1"));
  auto out_name = op_desc.Output("Y").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void TransposeOpMapper(const paddle::cpp::OpDesc& op_desc,
                       const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of transpose op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The output of transpose op must be 1"));
  auto out_name = op_desc.Output("Y").front();

  auto x = ctx.GetVar(x_name);

  PADDLE_ENFORCE_EQ(x->shape.size() == 2,
                    true,
                    phi::errors::InvalidArgument(
                        "Now transpose_p only support 2-dim matrix."));
  VLOG(4) << "Transpose " << x_name << " with shape ("
          << cinn::utils::Join(x->shape, ",") << ").";

  auto out = ctx.Builder()->Transpose(x, {1, 0});

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SliceSelectOpMapper(const paddle::cpp::OpDesc& op_desc,
                         const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of slice_select op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The output of slice_select op must be 1"));
  auto out_name = op_desc.Output("Y").front();

  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("starts"),
      true,
      phi::errors::InvalidArgument(
          "The slice_select_p operator should has 'starts' attribute."));
  auto starts = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "starts"));
  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("ends"),
      true,
      phi::errors::InvalidArgument(
          "The slice_select_p operator should has 'ends' attribute."));
  auto ends = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "ends"));
  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("axis"),
      true,
      phi::errors::InvalidArgument(
          "The slice_select_p operator should has 'axis' attribute."));
  auto axes = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "axis"));
  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("strides"),
      true,
      phi::errors::InvalidArgument(
          "The slice_select_p operator should has 'strides' attribute."));
  auto strides = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "strides"));

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "SliceSelect " << x_name << " from shape ("
          << cinn::utils::Join(x->shape, ",") << ") with starts ["
          << cinn::utils::Join(starts, ",") << "], ends ["
          << cinn::utils::Join(ends, ",") << "], axis ["
          << cinn::utils::Join(axes, ",") << "], strides ["
          << cinn::utils::Join(strides, ",") << "].";

  auto out = ctx.Builder()->Slice(x, axes, starts, ends, ShapeType{}, strides);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SliceAssignOpMapper(const paddle::cpp::OpDesc& op_desc,
                         const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of slice_assign op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The input of slice_assign op must be 1"));
  auto y_name = op_desc.Input("Y").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Z").size(),
      1UL,
      phi::errors::InvalidArgument("The output of slice_assign op must be 1"));
  auto out_name = op_desc.Output("Z").front();

  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("starts"),
      true,
      phi::errors::InvalidArgument(
          "The slice_assign_p operator should has 'starts' attribute."));
  auto starts = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "starts"));
  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("ends"),
      true,
      phi::errors::InvalidArgument(
          "The slice_assign_p operator should has 'ends' attribute."));
  auto ends = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "ends"));
  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("axis"),
      true,
      phi::errors::InvalidArgument(
          "The slice_assign_p operator should has 'axis' attribute."));
  auto axes = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "axis"));
  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("strides"),
      true,
      phi::errors::InvalidArgument(
          "The slice_assign_p operator should has 'strides' attribute."));
  auto strides = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "strides"));

  auto x = ctx.GetVar(x_name);
  auto assign = ctx.GetVar(y_name);

  VLOG(4) << "SliceAssign " << x_name << " from shape ("
          << cinn::utils::Join(x->shape, ",") << ") with starts ["
          << cinn::utils::Join(starts, ",") << "], ends ["
          << cinn::utils::Join(ends, ",") << "], axis ["
          << cinn::utils::Join(axes, ",") << "], strides ["
          << cinn::utils::Join(strides, ",") << "].";

  absl::optional<Variable> out;
  if (x->shape == assign->shape) {
    out = ctx.Builder()->Identity(assign);
  } else {
    out = ctx.Builder()->SliceAssign(x, assign, axes, starts, ends, strides);
  }

  ctx.AddVar(out_name, out.value());
  ctx.AddVarModelToProgram(out_name, out.value()->id);
}

void ReduceOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx,
                    const std::string& reduce_type) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of reduce op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The output of reduce op must be 1"));
  auto out_name = op_desc.Output("Y").front();

  auto axis = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "axis"));
  auto keepdim = utils::GetAttrOrDefault<bool>(op_desc, "keepdim", false);

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Reduce " << reduce_type << " x:" << x_name << " from shape ("
          << cinn::utils::Join(x->shape, ",") << "), with axis ["
          << cinn::utils::Join(axis, ",") << "], keepdim " << keepdim;

  // now paddle science only need reduce sum
  absl::optional<Variable> out;
  if (reduce_type == "Sum") {
    out = ctx.Builder()->ReduceSum(x, axis, keepdim);
  } else if (reduce_type == "Prod") {
    out = ctx.Builder()->ReduceProd(x, axis, keepdim);
  } else if (reduce_type == "Max") {
    out = ctx.Builder()->ReduceMax(x, axis, keepdim);
  } else if (reduce_type == "Min") {
    out = ctx.Builder()->ReduceMin(x, axis, keepdim);
  } else if (reduce_type == "All") {
    out = ctx.Builder()->ReduceAll(x, axis, keepdim);
  } else if (reduce_type == "Any") {
    out = ctx.Builder()->ReduceAny(x, axis, keepdim);
  }

  CHECK(out) << "Not support Reduce " << reduce_type << "! Please check.";

  ctx.AddVar(out_name, out.value());
  ctx.AddVarModelToProgram(out_name, out.value()->id);
}

#define EXPAND_REDUCE_OPMAPPER(ReduceType)                              \
  void Reduce##ReduceType##OpMapper(const paddle::cpp::OpDesc& op_desc, \
                                    const OpMapperContext& ctx) {       \
    ReduceOpMapper(op_desc, ctx, #ReduceType);                          \
  }

EXPAND_REDUCE_OPMAPPER(Sum)
EXPAND_REDUCE_OPMAPPER(Prod)
EXPAND_REDUCE_OPMAPPER(Max)
EXPAND_REDUCE_OPMAPPER(Min)
EXPAND_REDUCE_OPMAPPER(All)
EXPAND_REDUCE_OPMAPPER(Any)
#undef EXPAND_REDUCE_OPMAPPER

void GatherOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of gather op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("IndexTensor").size(),
      1UL,
      phi::errors::InvalidArgument("The input of gather op must be 1"));
  auto index_name = op_desc.Input("IndexTensor").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The output of gather op must be 1"));
  auto out_name = op_desc.Output("Y").front();

  auto axis =
      utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

  auto x = ctx.GetVar(x_name);
  auto index = ctx.GetVar(index_name);

  VLOG(4) << "Gather " << index_name << " ("
          << cinn::utils::Join(index->shape, ",") << ") from " << x_name
          << " shape (" << cinn::utils::Join(x->shape, ",") << ") "
          << "at dimension " << axis;

  auto out = ctx.Builder()->Gather(x, index, axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void IndexAssignOpMapper(const paddle::cpp::OpDesc& op_desc,
                         const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of index_assign op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The input of index_assign op must be 1"));
  auto updates_name = op_desc.Input("Y").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("IndexTensor").size(),
      1UL,
      phi::errors::InvalidArgument("The input of index_assign op must be 1"));
  auto index_name = op_desc.Input("IndexTensor").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Z").size(),
      1UL,
      phi::errors::InvalidArgument("The output of index_assign op must be 1"));
  auto out_name = op_desc.Output("Z").front();

  auto axis =
      utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

  auto x = ctx.GetVar(x_name);
  auto updates = ctx.GetVar(updates_name);
  auto index = ctx.GetVar(index_name);

  auto out = ctx.Builder()->ScatterAssign(x, updates, index, axis);

  VLOG(4) << "IndexAssign " << updates_name << " ("
          << cinn::utils::Join(updates->shape, ",") << ") to " << x_name
          << " shape (" << cinn::utils::Join(x->shape, ",") << ") "
          << "at dimension " << axis;

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ScatterAddOpMapper(const paddle::cpp::OpDesc& op_desc,
                        const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of scatter_add op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The input of scatter_add op must be 1"));
  auto updates_name = op_desc.Input("Y").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("IndexTensor").size(),
      1UL,
      phi::errors::InvalidArgument("The input of scatter_add op must be 1"));
  auto index_name = op_desc.Input("IndexTensor").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Z").size(),
      1UL,
      phi::errors::InvalidArgument("The output of scatter_add op must be 1"));
  auto out_name = op_desc.Output("Z").front();

  auto axis =
      utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

  auto x = ctx.GetVar(x_name);
  auto updates = ctx.GetVar(updates_name);
  auto index = ctx.GetVar(index_name);

  auto out = ctx.Builder()->ScatterAdd(x, updates, index, axis);

  VLOG(4) << "ScatterAdd " << updates_name << " ("
          << cinn::utils::Join(updates->shape, ",") << ") to " << x_name
          << " shape (" << cinn::utils::Join(x->shape, ",") << ") "
          << "at dimension " << axis;

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void SelectOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Condition").size(),
      1UL,
      phi::errors::InvalidArgument("The input of select op must be 1"));
  auto cond_name = op_desc.Input("Condition").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of select op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The input of select op must be 1"));
  auto y_name = op_desc.Input("Y").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Z").size(),
      1UL,
      phi::errors::InvalidArgument("The output of select op must be 1"));
  auto out_name = op_desc.Output("Z").front();

  VLOG(4) << cond_name << " ? " << x_name << " : " << y_name;

  auto cond = ctx.GetVar(cond_name);
  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);
  auto out = ctx.Builder()->Select(cond, x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void CastOpMapper(const paddle::cpp::OpDesc& op_desc,
                  const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of cast op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The output of cast op must be 1"));
  auto out_name = op_desc.Output("Y").front();

  auto x = ctx.GetVar(x_name);

  auto dtype_id = utils::GetAttrOrDefault<int>(
      op_desc, "dtype", static_cast<int>(paddle::cpp::VarDescAPI::Type::FP32));
  auto dtype_pd = static_cast<paddle::cpp::VarDescAPI::Type>(dtype_id);
  auto dtype_cinn = utils::CppVarType2CommonType(dtype_pd);
  auto dtype = cinn::common::Type2Str(dtype_cinn);

  VLOG(4) << out_name << " = cast(" << x_name << ", dtype=" << dtype << ")";

  auto out = ctx.Builder()->Cast(x, dtype);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace science_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(science_transform) {
  CINN_REGISTER_OP_MAPPER(concat_p,
                          cinn::frontend::science_mappers::ConcatOpMapper)
  CINN_REGISTER_OP_MAPPER(split_p,
                          cinn::frontend::science_mappers::SplitOpMapper)
  CINN_REGISTER_OP_MAPPER(reshape_p,
                          cinn::frontend::science_mappers::ReshapeOpMapper)
  CINN_REGISTER_OP_MAPPER(transpose_p,
                          cinn::frontend::science_mappers::TransposeOpMapper)
  CINN_REGISTER_OP_MAPPER(slice_select_p,
                          cinn::frontend::science_mappers::SliceSelectOpMapper)
  CINN_REGISTER_OP_MAPPER(slice_assign_p,
                          cinn::frontend::science_mappers::SliceAssignOpMapper)
  CINN_REGISTER_OP_MAPPER(index_select_p,
                          cinn::frontend::science_mappers::GatherOpMapper)
  CINN_REGISTER_OP_MAPPER(gather_p,
                          cinn::frontend::science_mappers::GatherOpMapper)
  CINN_REGISTER_OP_MAPPER(index_assign_p,
                          cinn::frontend::science_mappers::IndexAssignOpMapper)
  CINN_REGISTER_OP_MAPPER(scatter_add_p,
                          cinn::frontend::science_mappers::ScatterAddOpMapper)
  CINN_REGISTER_OP_MAPPER(reduce_p,
                          cinn::frontend::science_mappers::ReduceSumOpMapper)
  CINN_REGISTER_OP_MAPPER(select_p,
                          cinn::frontend::science_mappers::SelectOpMapper)
  CINN_REGISTER_OP_MAPPER(cast_p, cinn::frontend::science_mappers::CastOpMapper)

#define EXPAND_REDUCE_OP_MAPPER_REGISTER(op_name, ReduceType) \
  CINN_REGISTER_OP_MAPPER(                                    \
      op_name, cinn::frontend::science_mappers::Reduce##ReduceType##OpMapper)

  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_sum_p, Sum)
  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_prod_p, Prod)
  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_max_p, Max)
  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_min_p, Min)
  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_all_p, All)
  EXPAND_REDUCE_OP_MAPPER_REGISTER(reduce_any_p, Any)
#undef EXPAND_REDUCE_OP_MAPPER_REGISTER

  return true;
}
