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

#include <absl/types/optional.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/cinn/frontend/var_type_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void AssignOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Identity(x);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ShapeOpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("Input").size(), 1UL);
  auto x_name = op_desc.Input("Input").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Constant(x->shape,
                                     cinn::utils::TransValidVarName(out_name));

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void FillConstantOpMapper(const paddle::cpp::OpDesc& op_desc,
                          const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto y_name = op_desc.Output("Out").front();

  const auto& cinn_name = cinn::utils::TransValidVarName(y_name);
  CheckVarNameValid(cinn_name);

  auto shape = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "shape"));
  auto value = utils::GetAttrOrDefault<float>(op_desc, "value", 0.0f);
  auto str_value =
      utils::GetAttrOrDefault<std::string>(op_desc, "str_value", "");
  auto force_cpu = utils::GetAttrOrDefault<bool>(op_desc, "force_cpu", false);

  auto dtype = utils::GetPaddleDtype(
      op_desc, "dtype", paddle::cpp::VarDescAPI::Type::FP32);
  CHECK(!dtype.empty()) << "The op \"fill_constant\"'s attribute \"dtype\" "
                           "should not be unknown type! Please check.";

  absl::optional<Variable> out;
  if (op_desc.HasInput("ValueTensor") &&
      !op_desc.Input("ValueTensor").empty()) {
    CHECK_EQ(op_desc.Input("ValueTensor").size(), 1UL);
    auto value_name = op_desc.Input("ValueTensor").front();
    auto value_tensor = ctx.GetVar(value_name);

    VLOG(4) << "fill constant " << value_name << "=" << value_tensor
            << " with shape (" << cinn::utils::Join(shape, ",")
            << ") and dtype [" << dtype << "]";

    CHECK(value_tensor->shape == cinn::utils::ShapeType{1})
        << "The shape of [ValueTensor] should be [1], but here ["
        << cinn::utils::Join(value_tensor->shape, ", ") << "]";
    if (common::Type2Str(value_tensor->type) != dtype) {
      value_tensor = ctx.Builder()->Cast(value_tensor, dtype);
    }
    out = ctx.Builder()->BroadcastTo(value_tensor, shape);
    out.value().set_id(cinn_name);
  } else {
    if (!str_value.empty()) {
      VLOG(4) << "fill constant (" << str_value << ") with shape ("
              << cinn::utils::Join(shape, ",") << ") and dtype [" << dtype
              << "]";
      out = ctx.Builder()->FillConstant(
          shape, str_value, cinn_name, dtype, force_cpu);
    } else {
      VLOG(4) << "fill constant (" << value << ") with shape ("
              << cinn::utils::Join(shape, ",") << ") and dtype [" << dtype
              << "]";
      out = ctx.Builder()->FillConstant(
          shape, value, cinn_name, dtype, force_cpu);
    }
  }

  ctx.AddVar(y_name, out.value());
  ctx.AddVarModelToProgram(y_name, out.value()->id);
}

void FillAnyLikeOpMapper(const paddle::cpp::OpDesc& op_desc,
                         const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  auto x = ctx.GetVar(x_name);

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto y_name = op_desc.Output("Out").front();

  auto shape = utils::ToShapeType(x->shape);
  auto value = utils::GetAttrOrDefault<float>(op_desc, "value");

  auto dtype = utils::GetPaddleDtype(
      op_desc, "dtype", paddle::cpp::VarDescAPI::Type::FP32);
  if (dtype.empty()) {
    dtype = common::Type2Str(x->type);
  }

  VLOG(4) << "FillAnyLikeOp: fill constant (" << value << ") with shape ("
          << cinn::utils::Join(shape, ", ") << ") and dtype [" << dtype << "]";

  const auto& cinn_name = cinn::utils::TransValidVarName(y_name);
  CheckVarNameValid(cinn_name);

  auto out = ctx.Builder()->FillConstant(shape, value, cinn_name, dtype);

  ctx.AddVar(y_name, out);
  ctx.AddVarModelToProgram(y_name, out->id);
}

template <typename T>
std::pair<bool, T> IsArithmeticSequence(const std::vector<T>& vec) {
  if (vec.size() <= 1UL || (vec[1] - vec[0]) == 0) {
    return {false, static_cast<T>(0)};
  }

  auto first_diff = vec[1] - vec[0];
  for (int i = 2; i < vec.size(); ++i) {
    if ((vec[i] - vec[i - 1]) != first_diff) {
      return {false, first_diff};
    }
  }
  return {true, first_diff};
}

void AssignValueOpMapper(const paddle::cpp::OpDesc& op_desc,
                         const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  const auto& cinn_out_name = cinn::utils::TransValidVarName(out_name);

  const auto& bool_values_tmp =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "bool_values");
  std::vector<bool> bool_values;
  if (!bool_values_tmp.empty()) {
    std::transform(bool_values_tmp.begin(),
                   bool_values_tmp.end(),
                   std::back_inserter(bool_values),
                   [](int x) { return static_cast<bool>(x); });
  }
  const auto& fp32_values =
      utils::GetAttrOrDefault<std::vector<float>>(op_desc, "fp32_values");
  const auto& int32_values =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "int32_values");
  const auto& int64_values =
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "int64_values");

  absl::optional<Variable> out;
  if (!bool_values.empty()) {
    VLOG(4) << "The input of assign_value is ["
            << cinn::utils::Join(bool_values, ", ") << "]";

    out = ctx.Builder()->Constant(bool_values, cinn_out_name);
  } else if (!fp32_values.empty()) {
    VLOG(4) << "The input of assign_value is ["
            << cinn::utils::Join(fp32_values, ", ") << "]";

    auto adj_diff = IsArithmeticSequence(fp32_values);

    if (adj_diff.first) {
      VLOG(4) << "The input of assign_value is a arithmetic sequence. Using "
                 "Arange instead of Constant.";
      auto epsilone = adj_diff.second > 0
                          ? std::numeric_limits<float>::epsilon()
                          : -std::numeric_limits<float>::epsilon();

      out = ctx.Builder()->Arange(fp32_values.front(),
                                  fp32_values.back() + epsilone,
                                  adj_diff.second,
                                  "float32");
    } else {
      out = ctx.Builder()->Constant(fp32_values, cinn_out_name);
    }
  } else if (!int32_values.empty()) {
    VLOG(4) << "The input of assign_value is ["
            << cinn::utils::Join(int32_values, ", ") << "]";

    auto adj_diff = IsArithmeticSequence(int32_values);

    if (adj_diff.first) {
      VLOG(4) << "The input of assign_value is a arithmetic sequence. Using "
                 "Arange instead of Constant.";
      auto epsilone = adj_diff.second > 0 ? 1 : -1;

      out = ctx.Builder()->Arange(
          static_cast<float>(int32_values.front()),
          static_cast<float>(int32_values.back() + epsilone),
          static_cast<float>(adj_diff.second),
          "int32");
    } else {
      out = ctx.Builder()->Constant(int32_values, cinn_out_name);
    }
  } else if (!int64_values.empty()) {
    VLOG(4) << "The input of assign_value is ["
            << cinn::utils::Join(int64_values, ", ") << "]";

    auto adj_diff = IsArithmeticSequence(int64_values);

    if (adj_diff.first) {
      VLOG(4) << "The input of assign_value is a arithmetic sequence. Using "
                 "Arange instead of Constant.";
      auto epsilone = adj_diff.second > 0 ? 1 : -1;

      out = ctx.Builder()->Arange(
          static_cast<float>(int64_values.front()),
          static_cast<float>(int64_values.back() + epsilone),
          static_cast<float>(adj_diff.second),
          "int64");
    } else {
      out = ctx.Builder()->Constant(int64_values, cinn_out_name);
    }
  }

  CHECK(out) << "assign_value's input should not empty, but " << out_name
             << "not! Please check.";
  const auto& shape = utils::GetAttrOrDefault<std::vector<int>>(
      op_desc, "shape", out.value()->shape);
  if (shape != out.value()->shape) {
    out = ctx.Builder()->Reshape(out.value(), shape);
  }

  ctx.AddVar(out_name, out.value());
  ctx.AddVarModelToProgram(out_name, out.value()->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_constant) {
  CINN_REGISTER_OP_MAPPER(assign,
                          cinn::frontend::paddle_mappers::AssignOpMapper)
  CINN_REGISTER_OP_MAPPER(shape, cinn::frontend::paddle_mappers::ShapeOpMapper)
  CINN_REGISTER_OP_MAPPER(fill_constant,
                          cinn::frontend::paddle_mappers::FillConstantOpMapper)
  CINN_REGISTER_OP_MAPPER(fill_any_like,
                          cinn::frontend::paddle_mappers::FillAnyLikeOpMapper)
  CINN_REGISTER_OP_MAPPER(assign_value,
                          cinn::frontend::paddle_mappers::AssignValueOpMapper)

  return true;
}
