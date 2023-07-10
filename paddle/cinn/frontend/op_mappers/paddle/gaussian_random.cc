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
#include "paddle/cinn/frontend/var_type_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void GaussianRandomOpMapper(const paddle::cpp::OpDesc& op_desc,
                            const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto shape_origin =
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "shape");
  auto shape = utils::ToShapeType(shape_origin);

  auto mean = utils::GetAttrOrDefault<float>(op_desc, "mean", 0.0f);
  auto std = utils::GetAttrOrDefault<float>(op_desc, "std", 1.0f);
  auto seed = utils::GetAttrOrDefault<int>(op_desc, "seed", 0);

  auto dtype = utils::GetPaddleDtype(
      op_desc, "dtype", paddle::cpp::VarDescAPI::Type::FP32);
  CHECK(!dtype.empty()) << "The op \"gaussian_random\"'s attribute \"dtype\" "
                           "should not be unknown type! Please check.";

  VLOG(4) << out_name << "[" << cinn::utils::Join(shape, ", ")
          << "] = uniform_random(mean=" << mean << ", std=" << std
          << ", seed=" << seed << ", dtype=" << dtype << ", shape=["
          << cinn::utils::Join(shape, ", ") << "])";

  auto out = ctx.Builder()->GaussianRandom(shape, mean, std, seed, dtype);
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_gaussian_random) {
  CINN_REGISTER_OP_MAPPER(
      gaussian_random, cinn::frontend::paddle_mappers::GaussianRandomOpMapper)
  return true;
}
