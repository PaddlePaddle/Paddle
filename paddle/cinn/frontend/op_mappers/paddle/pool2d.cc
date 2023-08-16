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

void Pool2dOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  CHECK(op_desc.HasAttr("pooling_type"));
  auto pooling_type =
      utils::GetAttrOrDefault<std::string>(op_desc, "pooling_type");
  CHECK(op_desc.HasAttr("ksize"));
  auto ksize = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "ksize");

  auto strides =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto padding_size =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});

  auto ceil_mode = utils::GetAttrOrDefault<bool>(op_desc, "ceil_mode", false);
  auto exclusive = utils::GetAttrOrDefault<bool>(op_desc, "exclusive", true);
  auto global_pooling =
      utils::GetAttrOrDefault<bool>(op_desc, "global_pooling", false);
  auto data_format =
      utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "NCHW");
  auto adaptive = utils::GetAttrOrDefault<bool>(op_desc, "adaptive", false);
  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(
      op_desc, "padding_algorithm", "EXPLICIT");
  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Pool2d(x,
                                   pooling_type,
                                   ksize,
                                   strides,
                                   padding_size,
                                   ceil_mode,
                                   exclusive,
                                   global_pooling,
                                   data_format,
                                   adaptive,
                                   padding_algorithm);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void Pool2dGradOpMapper(const paddle::cpp::OpDesc& op_desc,
                        const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Out").size(), 1UL);
  auto y_name = op_desc.Input("Out").front();
  CHECK_EQ(op_desc.Input(paddle::GradVarName("Out")).size(), 1UL);
  auto dy_name = op_desc.Input(paddle::GradVarName("Out")).front();

  CHECK_EQ(op_desc.Output(paddle::GradVarName("X")).size(), 1UL);
  auto dx_name = op_desc.Output(paddle::GradVarName("X")).front();

  CHECK(op_desc.HasAttr("pooling_type"));
  auto pooling_type =
      utils::GetAttrOrDefault<std::string>(op_desc, "pooling_type");
  CHECK(op_desc.HasAttr("ksize"));
  auto ksize = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "ksize");

  auto strides =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto padding_size =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});

  auto ceil_mode = utils::GetAttrOrDefault<bool>(op_desc, "ceil_mode", false);
  auto exclusive = utils::GetAttrOrDefault<bool>(op_desc, "exclusive", true);
  auto global_pooling =
      utils::GetAttrOrDefault<bool>(op_desc, "global_pooling", false);
  auto data_format =
      utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "NCHW");
  auto adaptive = utils::GetAttrOrDefault<bool>(op_desc, "adaptive", false);
  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(
      op_desc, "padding_algorithm", "EXPLICIT");

  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);
  auto dy = ctx.GetVar(dy_name);

  auto out = ctx.Builder()->Pool2dGrad(x,
                                       y,
                                       dy,
                                       pooling_type,
                                       ksize,
                                       strides,
                                       padding_size,
                                       ceil_mode,
                                       exclusive,
                                       global_pooling,
                                       data_format,
                                       adaptive,
                                       padding_algorithm);

  ctx.AddVar(dx_name, out);
  ctx.AddVarModelToProgram(dx_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_pool2d) {
  CINN_REGISTER_OP_MAPPER(pool2d,
                          cinn::frontend::paddle_mappers::Pool2dOpMapper)
  CINN_REGISTER_OP_MAPPER(pool2d_grad,
                          cinn::frontend::paddle_mappers::Pool2dGradOpMapper)
  return true;
}
