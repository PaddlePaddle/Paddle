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
#pragma once

#include <limits.h>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace cinn {
namespace dialect {
namespace ir {
// alias OpPatternKind and pir::Group
using OpPatternKind = hlir::framework::OpPatternKind;

template <typename T = int64_t>
std::vector<T> GetVectorAttr(const ::pir::Operation* op,
                             const std::string& name) {
  auto& attr_map = op->attributes();
  PADDLE_ENFORCE(
      attr_map.count(name),
      ::common::errors::PreconditionNotMet(
          "attr [%s] MUST in attribute map for [%s] op", name, op->name()));
  auto& val = attr_map.at(name);

  PADDLE_ENFORCE(val.isa<::pir::ArrayAttribute>(),
                 ::common::errors::PreconditionNotMet(
                     "axis Type MUST ArrayAttribute for [%s] op", op->name()));
  auto array_list = val.dyn_cast<::pir::ArrayAttribute>().AsVector();
  std::vector<T> vec_res;
  if (array_list.size() > 0) {
    PADDLE_ENFORCE_EQ(array_list[0].isa<::pir::Int64Attribute>(),
                      true,
                      ::common::errors::Unimplemented(
                          "the 0th elementwise MUST be ir::Int64Attribute"));
    for (size_t i = 0; i < array_list.size(); ++i) {
      vec_res.push_back(array_list[i].dyn_cast<::pir::Int64Attribute>().data());
    }
  }
  return vec_res;
}

phi::DDim GetFirstInputShape(const ::pir::Operation* op);

const phi::DDim& GetValueShape(const ::pir::Value& value);

bool WithoutLastDimInReduce(const std::vector<int64_t>& inshape,
                            const std::vector<int64_t>& axes);

int GetSharedSize(::pir::Operation* op);

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
