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

#include <numeric>

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void GatherOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Index").size(), 1UL);
  auto index_name = op_desc.Input("Index").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x = ctx.GetVar(x_name);
  auto index = ctx.GetVar(index_name);

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", 0);

  VLOG(4) << "Gather X:" << x_name << "[" << cinn::utils::Join(x->shape, ",")
          << "] with index:" << index_name << "["
          << cinn::utils::Join(index->shape, ",") << "] at axis=" << axis;

  if (index->shape.size() > 1) {
    // reshape index if the rank of index is greater than 1
    bool is_rank_1 = false;
    for (auto dim : index->shape) {
      if (dim != 1) {
        CHECK(!is_rank_1) << "The \"index\" of \"Gather\" only support rank 1 "
                             "tensor, but here index.shape=["
                          << cinn::utils::Join(index->shape, ",") << "]";
        is_rank_1 = true;
      }
    }
    auto num = std::accumulate(
        index->shape.begin(), index->shape.end(), 1, std::multiplies<int>());
    index = ctx.Builder()->Reshape(index, {num});
  }

  // now paddle science only need reduce sum
  auto out = ctx.Builder()->Gather(x, index, axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_gather) {
  CINN_REGISTER_OP_MAPPER(gather,
                          cinn::frontend::paddle_mappers::GatherOpMapper)
  return true;
}
