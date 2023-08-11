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

#include <string>

#include "paddle/cinn/common/context.h"
#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void LayerNormOpMapper(const paddle::cpp::OpDesc& op_desc,
                       const OpMapperContext& ctx) {
  auto get_input = [&op_desc](const std::string& name) {
    CHECK_EQ(op_desc.Input(name).size(), 1UL);
    return op_desc.Input(name).front();
  };
  auto get_output = [&op_desc](const std::string& name) {
    CHECK_EQ(op_desc.Output(name).size(), 1UL);
    return op_desc.Output(name).front();
  };

  // get input names
  auto x_name = get_input("X");
  absl::optional<std::string> scale_name;
  if (op_desc.HasInput("Scale")) {
    scale_name = get_input("Scale");
  }
  absl::optional<std::string> bias_name;
  if (op_desc.HasInput("Bias")) {
    bias_name = get_input("Bias");
  }
  // get attribute values
  auto epsilon = utils::GetAttrOrDefault<float>(op_desc, "epsilon", 1e-5f);
  auto begin_norm_axis =
      utils::GetAttrOrDefault<int>(op_desc, "begin_norm_axis", 1);
  // get input variable
  auto x = ctx.GetVar(x_name);
  absl::optional<Variable> scale;
  if (scale_name) {
    scale = ctx.GetVar(*scale_name);
  }
  absl::optional<Variable> bias;
  if (bias_name) {
    bias = ctx.GetVar(*bias_name);
  }

  VLOG(4) << "layer_norm X=" << x_name << "[" << x
          << "], Scale=" << scale_name.value() << "[" << scale.value()
          << "], Bias=" << bias_name.value() << "[" << bias.value()
          << "], epsilon=" << epsilon
          << ", begin_norm_axis=" << begin_norm_axis;

  const auto& x_shape = x->shape;
  auto x_ndim = x_shape.size();
  CHECK_LT(begin_norm_axis, x_ndim) << "`begin_norm_axis` must be less than "
                                       "the dimensions of X, but received "
                                    << begin_norm_axis;
  VLOG(4) << "-- [layer_norm] begin_norm_axis = " << begin_norm_axis;
  int left = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    left *= x_shape[i];
  }
  int right = 1;
  for (int i = begin_norm_axis; i < x_ndim; i++) {
    right *= x_shape[i];
  }
  VLOG(4) << "-- [layer_norm] left = " << left << ", right = " << right;

  // compute mean
  auto* builder = ctx.Builder();

  const auto& x_type = x->type;
  if (x_type.is_float16() || x_type.is_bfloat16()) {
    x = builder->Cast(x, "float32");
  }

  std::vector<int> shape{left, right};
  auto x_reshape = builder->Reshape(x, shape);
  auto x_reduce = builder->ReduceSum(x_reshape, {1});
  auto ele_num = builder->FillConstant({left},
                                       static_cast<float>(right),
                                       common::UniqName("layer_norm_ele_num"),
                                       common::Type2Str(x->type));
  auto x_mean = builder->Divide(x_reduce, ele_num);

  // use `E[|x|^2] - |E[x]|^2` instead of `E[|x - E[x]|^2])` to compute variance
  auto x2 = builder->Multiply(x_reshape, builder->Identity(x_reshape));
  auto x2_reduce = builder->ReduceSum(x2, {1});
  auto x2_mean = builder->Divide(x2_reduce, ele_num);
  auto x_mean2 = builder->Multiply(x_mean, builder->Identity(x_mean));
  auto zero = builder->FillConstant({left},
                                    0.f,
                                    common::UniqName("layer_norm_zero"),
                                    common::Type2Str(x->type));
  auto x_var = builder->Max(builder->Subtract(x2_mean, x_mean2), zero);

  // compute x norm
  auto x_mean_broadcast = builder->BroadcastTo(x_mean, shape, {0});
  auto y_sub = builder->Subtract(x_reshape, x_mean_broadcast);
  auto epsilon_var =
      builder->FillConstant({left},
                            epsilon,
                            common::UniqName("layer_norm_epsilon"),
                            common::Type2Str(x->type));
  auto x_var_eps = builder->Add(x_var, epsilon_var);
  auto x_var_sqrt = builder->Sqrt(x_var_eps);
  auto y_out =
      builder->Divide(y_sub, builder->BroadcastTo(x_var_sqrt, shape, {0}));

  // multiply scale
  if (scale) {
    if (scale.value()->type.is_float16() || scale.value()->type.is_bfloat16()) {
      scale = ctx.Builder()->Cast(scale.value(), "float32");
    }
    auto scale_broadcast = builder->BroadcastTo(*scale, shape, {1});
    y_out = builder->Multiply(y_out, scale_broadcast);
  }

  // add bias
  if (bias) {
    if (bias.value()->type.is_float16() || bias.value()->type.is_bfloat16()) {
      bias = ctx.Builder()->Cast(bias.value(), "float32");
    }
    auto bias_broadcast = builder->BroadcastTo(*bias, shape, {1});
    y_out = builder->Add(y_out, bias_broadcast);
  }

  // reshape to the original shape
  y_out = builder->Reshape(y_out, x_shape);

  if (x_type.is_float16()) {
    y_out = builder->Cast(y_out, "float16");
  } else if (x_type.is_bfloat16()) {
    y_out = builder->Cast(y_out, "bfloat16");
  }

  // get output names
  auto y_name = get_output("Y");
  auto mean_name = get_output("Mean");
  auto variance_name = get_output("Variance");
  // re-mapper outputs
  ctx.AddVar(y_name, y_out);
  ctx.AddVarModelToProgram(y_name, y_out->id);
  ctx.AddVar(mean_name, x_mean);
  ctx.AddVarModelToProgram(mean_name, x_mean->id);
  ctx.AddVar(variance_name, x_var);
  ctx.AddVarModelToProgram(variance_name, x_var->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_layer_norm) {
  CINN_REGISTER_OP_MAPPER(layer_norm,
                          cinn::frontend::paddle_mappers::LayerNormOpMapper)
  return true;
}
