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

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ExpandOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  CHECK(op_desc.HasAttr("expand_times"));
  auto expand_times =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "expand_times");

  auto x = ctx.GetVar(x_name);
  auto x_shape = x->shape;

  VLOG(4) << "expand: x shape: " << cinn::utils::Join(x_shape, ", ");
  VLOG(4) << "expand: attr expand_times: "
          << cinn::utils::Join(expand_times, ", ");

  CHECK_EQ(expand_times.size(), x_shape.size())
      << "The size of `expand_times' should == the rank[" << x_shape.size()
      << "] of x's shape, but got " << expand_times.size();

  std::vector<int> out_shape(x_shape.size());
  for (size_t i = 0; i < x_shape.size(); ++i) {
    out_shape[i] = x_shape[i] * expand_times[i];
  }

  VLOG(4) << "expand: out shape: " << cinn::utils::Join(out_shape, ", ");

  auto out = ctx.Builder()->BroadcastTo(x, out_shape);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void ExpandV2OpMapper(const paddle::cpp::OpDesc& op_desc,
                      const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  CHECK(op_desc.HasAttr("shape"));
  auto shape = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "shape");

  auto x = ctx.GetVar(x_name);
  auto x_shape = x->shape;

  VLOG(4) << "expand_v2: x shape: " << cinn::utils::Join(x_shape, ", ");
  VLOG(4) << "expand_v2: attr shape: " << cinn::utils::Join(shape, ", ");

  CHECK_GE(shape.size(), x_shape.size())
      << "The size of `shape' should >= the rank[" << x_shape.size()
      << "] of x's shape, but got " << shape.size();

  auto diff = shape.size() - x_shape.size();
  x_shape.insert(x_shape.begin(), diff, 1);

  std::vector<int> out_shape(x_shape.size());

  for (size_t i = 0; i < x_shape.size(); ++i) {
    CHECK_NE(shape[i], 0) << "The " << i
                          << "th element in shape cannot be zero.";
    if (i < diff) {
      CHECK_GT(shape[i], 0)
          << "The " << i << "th element[" << shape[i]
          << "] for non-existing dimensions must be positive.";
      out_shape[i] = shape[i];
    } else if (shape[i] > 0) {
      if (x_shape[i] != 1) {
        CHECK_EQ(shape[i], x_shape[i])
            << "The " << i << "th element[" << shape[i]
            << "] of the non-singleton dimension does not match"
               " the corresponding element["
            << x_shape[i] << "] in x's shape.";
        out_shape[i] = shape[i];
      } else {
        out_shape[i] = shape[i];
      }
    } else {
      CHECK_EQ(shape[i], -1) << "When the element in shape is negative for "
                                "expand_v2 op, only -1 is supported, but got "
                             << shape[i];
      out_shape[i] = x_shape[i];
    }
  }

  VLOG(4) << "expand_v2: out shape: " << cinn::utils::Join(out_shape, ", ");

  Variable out;
  if (out_shape == x_shape) {
    out = ctx.Builder()->Identity(x);
  } else {
    out = ctx.Builder()->BroadcastTo(x, out_shape);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_expand) {
  CINN_REGISTER_OP_MAPPER(expand,
                          cinn::frontend::paddle_mappers::ExpandOpMapper)
  CINN_REGISTER_OP_MAPPER(expand_v2,
                          cinn::frontend::paddle_mappers::ExpandV2OpMapper)
  return true;
}
