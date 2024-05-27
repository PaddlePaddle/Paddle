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

void SoftmaxOpMapper(const paddle::cpp::OpDesc& op_desc,
                     const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of Softmax op must be 1."));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Out").size(),
      1UL,
      phi::errors::InvalidArgument("The output of Softmax op must be 1."));
  auto out_name = op_desc.Output("Out").front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);
  auto data_format =
      utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "AnyLayout");

  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Softmax(x, {axis}, data_format);
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_softmax) {
  CINN_REGISTER_OP_MAPPER(softmax,
                          cinn::frontend::paddle_mappers::SoftmaxOpMapper)
  return true;
}
