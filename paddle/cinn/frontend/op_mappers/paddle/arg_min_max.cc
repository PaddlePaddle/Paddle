// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/frontend/var_type_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

enum class ArgType { ArgMax, ArgMin };

template <ArgType type>
Variable ArgImpl(NetBuilder* builder,
                 const Variable& x,
                 int axis,
                 bool keepdims);

template <>
Variable ArgImpl<ArgType::ArgMax>(NetBuilder* builder,
                                  const Variable& x,
                                  int axis,
                                  bool keepdims) {
  return builder->Argmax(x, axis, keepdims);
}

template <>
Variable ArgImpl<ArgType::ArgMin>(NetBuilder* builder,
                                  const Variable& x,
                                  int axis,
                                  bool keepdims) {
  return builder->Argmin(x, axis, keepdims);
}

template <ArgType type>
void ArgOpMapperHelper(const paddle::cpp::OpDesc& op_desc,
                       const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x = ctx.GetVar(x_name);
  auto axis = utils::GetAttrOrDefault<int64_t>(op_desc, "axis", -1);
  CHECK(op_desc.HasAttr("axis"))
      << "Argmax/Argmin op should has attribute \"axis\"! Please check.";

  auto keepdims = utils::GetAttrOrDefault<bool>(op_desc, "keepdims", false);
  CHECK(op_desc.HasAttr("keepdims"))
      << "Argmax/Argmin op should has attribute \"keepdims\"! Please check.";

  auto flatten = utils::GetAttrOrDefault<bool>(op_desc, "flatten", false);
  CHECK(op_desc.HasAttr("flatten"))
      << "Argmax/Argmin op should has attribute \"flatten\"! Please check.";

  auto dtype = utils::GetPaddleDtype(
      op_desc, "dtype", paddle::cpp::VarDescAPI::Type::INT64);
  CHECK(dtype == "int32" || dtype == "int64")
      << "the indices dtype must be int32 or int64, but got dtype = " << dtype;

  int ndim = x->shape.size();
  // If flatten = true, flatten x and do opration on axis 0.
  if (flatten) {
    x = ctx.Builder()->Reshape(x, {-1});
    axis = 0;
    ndim = x->shape.size();
  }

  auto out = ArgImpl<type>(ctx.Builder(), x, axis, keepdims);

  out = ctx.Builder()->Cast(out, dtype);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ArgMaxOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  ArgOpMapperHelper<ArgType::ArgMax>(op_desc, ctx);
}

void ArgMinOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  ArgOpMapperHelper<ArgType::ArgMin>(op_desc, ctx);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_arg) {
  CINN_REGISTER_OP_MAPPER(arg_max,
                          cinn::frontend::paddle_mappers::ArgMaxOpMapper)
  CINN_REGISTER_OP_MAPPER(arg_min,
                          cinn::frontend::paddle_mappers::ArgMinOpMapper)

  return true;
}
