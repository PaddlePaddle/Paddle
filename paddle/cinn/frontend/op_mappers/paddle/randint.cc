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
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace paddle_mappers {

void RandIntOpMapper(const paddle::cpp::OpDesc& op_desc,
                     const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Out").size(),
      1UL,
      phi::errors::InvalidArgument("The output of randint op should be one."));
  auto out_name = op_desc.Output("Out").front();

  PADDLE_ENFORCE_EQ(op_desc.HasAttr("shape"),
                    true,
                    phi::errors::InvalidArgument(
                        "The randint op should have shape attribute."));

  auto shape_origin =
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "shape");
  auto shape = utils::ToShapeType(shape_origin);

  PADDLE_ENFORCE_EQ(op_desc.HasAttr("low"),
                    true,
                    phi::errors::InvalidArgument(
                        "The randint op should have low attribute."));

  auto min = utils::GetAttrOrDefault<int>(op_desc, "low", 0);

  PADDLE_ENFORCE_EQ(op_desc.HasAttr("high"),
                    true,
                    phi::errors::InvalidArgument(
                        "The randint op should have high attribute."));

  auto max = utils::GetAttrOrDefault<int>(op_desc, "high", 0);
  PADDLE_ENFORCE_GT(max,
                    min,
                    phi::errors::InvalidArgument(
                        "max should greater than min! Please check."));

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
