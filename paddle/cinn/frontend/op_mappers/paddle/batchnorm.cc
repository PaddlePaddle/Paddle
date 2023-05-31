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

void BatchNormOpMapper(const paddle::cpp::OpDesc& op_desc,
                       const OpMapperContext& ctx) {
  auto add_output = [&op_desc, &ctx](const std::string& pd_param_name,
                                     const Variable& out,
                                     bool can_inplace = false) -> void {
    if (!op_desc.HasOutput(pd_param_name)) {
      VLOG(4) << "Cannot find parameter " << pd_param_name << " in op "
              << op_desc.Type();
      return;
    }
    CHECK_EQ(op_desc.Output(pd_param_name).size(), 1UL);
    auto output_name = op_desc.Output(pd_param_name).front();

    VLOG(4) << "The " << op_desc.Type() << "'s output " << pd_param_name
            << " is " << output_name;

    ctx.AddVar(output_name, out, can_inplace);
    ctx.AddVarModelToProgram(output_name, out->id, can_inplace);
  };

  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Scale").size(), 1UL);
  auto scale_name = op_desc.Input("Scale").front();
  CHECK_EQ(op_desc.Input("Bias").size(), 1UL);
  auto bias_name = op_desc.Input("Bias").front();
  CHECK_EQ(op_desc.Input("Mean").size(), 1UL);
  auto mean_name = op_desc.Input("Mean").front();
  CHECK_EQ(op_desc.Input("Variance").size(), 1UL);
  auto variance_name = op_desc.Input("Variance").front();

  auto epsilon = utils::GetAttrOrDefault<float>(op_desc, "epsilon", 1e-5f);
  auto momentum = utils::GetAttrOrDefault<float>(op_desc, "momentum", 0.9f);
  auto data_layout =
      utils::GetAttrOrDefault<std::string>(op_desc, "data_layout", "NCHW");
  auto x = ctx.GetVar(x_name);
  auto scale = ctx.GetVar(scale_name);
  auto bias = ctx.GetVar(bias_name);
  auto mean = ctx.GetVar(mean_name);
  auto variance = ctx.GetVar(variance_name);

  auto is_test = utils::GetAttrOrDefault<bool>(op_desc, "is_test", false);
  auto trainable_stats =
      utils::GetAttrOrDefault<bool>(op_desc, "trainable_statistics", false);
  auto use_global_stats =
      utils::GetAttrOrDefault<bool>(op_desc, "use_global_stats", false);
  bool use_run_stat = (is_test && (!trainable_stats)) || use_global_stats;

  VLOG(4) << "Try compute batch_norm(X:" << x_name << ", Scale:" << scale_name
          << ", Bias:" << bias_name
          << ","
             ", Mean:"
          << mean_name << ", Variance:" << variance_name
          << ", epsilon=" << epsilon << ", momentum=" << momentum
          << ", data_layout=" << data_layout << ", is_test=" << is_test
          << ", trainable_statistics=" << trainable_stats
          << ", use_global_stats=" << use_global_stats << ")";

  auto outs = ctx.Builder()->BatchNorm(x,
                                       scale,
                                       bias,
                                       mean,
                                       variance,
                                       epsilon,
                                       momentum,
                                       data_layout,
                                       use_run_stat);

  if (use_run_stat) {
    VLOG(4) << "Invoke batch_norm OpMapper with test mode";

    add_output("Y", outs[0]);
    // batch_norm eval mode should not modify mean and variance's value
    auto save_mean = ctx.Builder()->Identity(mean);
    add_output("SavedMean", save_mean);
    auto save_variance = ctx.Builder()->Identity(variance);
    add_output("SavedVariance", save_variance);
    // Just for skip error of "Variable(batch_norm2d_0.w_2@InplaceOut) not
    // applied in cinn" when run batchnorm in paddle, remove after inpace
    // mechanism perfect. The value should shared memory with mean and variance.
    auto mean_out = ctx.Builder()->Identity(mean);
    add_output("MeanOut", mean_out, true);
    auto variance_out = ctx.Builder()->Identity(variance);
    add_output("VarianceOut", variance_out, true);
  } else {
    VLOG(4) << "Invoke batch_norm OpMapper with train mode";
    CHECK_EQ(outs.size(), 5U)
        << "batch_norm in train mode should only has 5 output! Please check.";

    add_output("Y", outs[0]);
    add_output("SavedMean", outs[1]);
    add_output("SavedVariance", outs[2]);
    // the argument of MeanOut and VarianceOut are the same as Mean and Variance
    add_output("MeanOut", outs[3], true);
    add_output("VarianceOut", outs[4], true);
  }
}

void BatchNormGradOpMapper(const paddle::cpp::OpDesc& op_desc,
                           const OpMapperContext& ctx) {
  std::unordered_map<std::string, std::string> input_names_map;
  auto get_input_var =
      [&op_desc, &ctx, &input_names_map](const std::string& op_name) {
        CHECK_EQ(op_desc.Input(op_name).size(), 1UL);
        auto var_name = op_desc.Input(op_name).front();
        input_names_map.emplace(op_name, var_name);
        return ctx.GetVar(var_name);
      };

  std::unordered_map<std::string, std::string> output_names_map;
  auto get_output_name =
      [&op_desc, &output_names_map](const std::string& op_name) -> std::string {
    if (op_desc.Output(op_name).empty()) {
      CHECK_NE(op_name, paddle::GradVarName("X"))
          << "The input X should not empty.";
      return "";
    }

    CHECK_EQ(op_desc.Output(op_name).size(), 1UL);
    auto var_name = op_desc.Output(op_name).front();
    output_names_map.emplace(op_name, var_name);
    return var_name;
  };

  std::vector<std::string> output_names = {
      get_output_name(paddle::GradVarName("X")),
      get_output_name(paddle::GradVarName("Scale")),
      get_output_name(paddle::GradVarName("Bias"))};

  auto x = get_input_var("X");
  auto dy = get_input_var(paddle::GradVarName("Y"));
  auto scale = get_input_var("Scale");
  auto saved_mean = get_input_var("SavedMean");
  auto saved_variance = get_input_var("SavedVariance");

  auto data_layout =
      utils::GetAttrOrDefault<std::string>(op_desc, "data_layout", "NCHW");
  auto epsilon = utils::GetAttrOrDefault<float>(op_desc, "epsilon", 1e-5f);

  auto get_arg_debug_info =
      [](const std::unordered_map<std::string, std::string>& names_map) {
        std::string res;
        for (const auto& pair : names_map) {
          res.append(pair.first + ":" + pair.second + ", ");
        }
        return res;
      };

  VLOG(4) << "{" << get_arg_debug_info(output_names_map)
          << "} = batch_norm_grad(" << get_arg_debug_info(input_names_map)
          << ", data_layout=" << data_layout << ", epsilon=" << epsilon << ")";

  // batch norm grad, output(grad_x, grad_scale, grad_bias)
  auto outs = ctx.Builder()->BatchNormGrad(
      dy, x, scale, saved_mean, saved_variance, epsilon, data_layout);
  CHECK_EQ(outs.size(), 3ul)
      << "batch_norm_grad APIs should return 3 Variable!";

  for (int i = 0; i < outs.size(); i++) {
    if (output_names[i].empty()) {
      continue;
    }

    ctx.AddVar(output_names[i], outs[i]);
    ctx.AddVarModelToProgram(output_names[i], outs[i]->id);
  }
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_batchnorm) {
  CINN_REGISTER_OP_MAPPER(batch_norm,
                          cinn::frontend::paddle_mappers::BatchNormOpMapper)
  CINN_REGISTER_OP_MAPPER(batch_norm_grad,
                          cinn::frontend::paddle_mappers::BatchNormGradOpMapper)
  return true;
}
