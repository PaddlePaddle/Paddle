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
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace paddle_mappers {

void GatherNdOpMapper(const paddle::cpp::OpDesc& op_desc,
                      const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of gather_nd op should be one."));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(op_desc.Input("Index").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "The input index of gather_nd op should be one."));
  auto index_name = op_desc.Input("Index").front();
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "The output of gather_nd op should be one."));
  auto out_name = op_desc.Output("Out").front();

  auto x = ctx.GetVar(x_name);
  auto index = ctx.GetVar(index_name);

  VLOG(4) << "GatherND X:" << x_name << "[" << cinn::utils::Join(x->shape, ",")
          << "] with index:" << index_name << "["
          << cinn::utils::Join(index->shape, ",") << "]";

  auto out = ctx.Builder()->GatherNd(x, index);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_gather_nd) {
  CINN_REGISTER_OP_MAPPER(gather_nd,
                          cinn::frontend::paddle_mappers::GatherNdOpMapper)
  return true;
}
