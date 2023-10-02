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

#include "glog/logging.h"
#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/cinn/frontend/var_type_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void RandIntOpMapper(const paddle::cpp::OpDesc& op_desc,
                     const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  CHECK(op_desc.HasAttr("shape")) << "Cannot find attribute \"shape\" in "
                                     "paddle op \"randint\"! Please check.";
  auto shape_origin =
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "shape");
  auto shape = utils::ToShapeType(shape_origin);

  CHECK(op_desc.HasAttr("low")) << "Cannot find attribute \"low\" in paddle op "
                                   "\"randint\"! Please check.";
  auto min = utils::GetAttrOrDefault<int>(op_desc, "low", 0);

  CHECK(op_desc.HasAttr("high")) << "Cannot find attribute \"high\" in paddle "
                                    "op \"randint\"! Please check.";
  auto max = utils::GetAttrOrDefault<int>(op_desc, "high", 0);
  CHECK_GT(max, min) << "max(" << max << ") should greater than min(" << min
                     << ")! Please check.";

  auto seed = utils::GetAttrOrDefault<int>(op_desc, "seed", 0);

  auto dtype = utils::GetPaddleDtype(
      op_desc, "dtype", paddle::cpp::VarDescAPI::Type::INT64);
  CHECK(dtype == "int32" || dtype == "int64")
      << "the indices dtype must be int32 or int64, but got dtype = " << dtype;

  auto out = ctx.Builder()->RandInt(shape, min, max, seed, dtype);
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_randint) {
  CINN_REGISTER_OP_MAPPER(randint,
                          cinn::frontend::paddle_mappers::RandIntOpMapper)
  return true;
}
