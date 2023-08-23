// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

void TileOpMapper(const paddle::cpp::OpDesc& op_desc,
                  const OpMapperContext& ctx) {
  // input
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  // output
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  // attr repeat_times
  std::vector<int> repeat_times =
      op_desc.GetAttr<std::vector<int>>("repeat_times");

  for (auto i : repeat_times) {
    CHECK_GT(i, 0) << "repeat_times's element must be greater than 0";
  }

  auto x = ctx.GetVar(x_name);

  // promotion
  auto vec_x_dims = std::vector<int>(x->shape);
  if (repeat_times.size() < vec_x_dims.size()) {
    int diff = vec_x_dims.size() - repeat_times.size();
    repeat_times.insert(repeat_times.begin(), diff, 1);
  } else {
    int diff = repeat_times.size() - vec_x_dims.size();
    vec_x_dims.insert(vec_x_dims.begin(), diff, 1);
  }

  CHECK_EQ(vec_x_dims.size(), repeat_times.size())
      << "vec_x_dims's size must be equal to repeat_times's size after "
         "promotion";

  // output's shape
  std::vector<int> output_shape = vec_x_dims;

  // calucate output's shape
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    output_shape[i] *= repeat_times[i];
  }

  VLOG(4) << "output_shape: " << cinn::utils::Join(output_shape, ",");

  // NOTE(wuweilong): Paddle's tile is implemented by Eigen's broadcast
  // directly, but CINN's tile can not be implemented by BroadcastTo directly,
  // because it is different from Eigen's broadcast. The semantics of Eigen's
  // broadcast is same as tile, but CINN can not use Eigen's broadcast. So we
  // need to Combine Reshape and BroadcastTo to implement tile.

  // make a copy of vec_x_dims
  std::vector<int> vec_x_dims_copy = vec_x_dims;
  // recontruct vec_x_dims_copy by inserting 1 before every element
  for (size_t i = 0; i < vec_x_dims_copy.size(); ++i) {
    vec_x_dims_copy.insert(vec_x_dims_copy.begin() + i, 1);
    i++;
  }

  x = ctx.Builder()->Reshape(x, vec_x_dims_copy);

  // recontruct vec_x_dims_copy for BroadaCast
  for (size_t i = 0; i < vec_x_dims_copy.size(); ++i) {
    if (i % 2 == 0) {
      vec_x_dims_copy[i] = output_shape[i / 2] / vec_x_dims_copy[i + 1];
    }
  }

  auto tmp = ctx.Builder()->BroadcastTo(x, vec_x_dims_copy);
  auto output = ctx.Builder()->Reshape(tmp, output_shape);

  ctx.AddVar(out_name, output);
  ctx.AddVarModelToProgram(out_name, output->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_tile) {
  CINN_REGISTER_OP_MAPPER(tile, cinn::frontend::paddle_mappers::TileOpMapper)
  return true;
}
