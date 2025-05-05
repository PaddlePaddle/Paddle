// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/paddle/pir/type_adt_type_id.h"
#include "paddle/ap/include/paddle/pir/type_method_class.h"

namespace ap::paddle {

adt::Result<axpr::Value> PirShapeOrDataString(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(self,
                    self_val.template CastTo<symbol::ShapeOrDataDimExprs>());
  std::ostringstream ss;
  ss << self;
  return ss.str();
}

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetPirShapeOrDataClass() {
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "PirShapeOrData",
      [&](const auto& Yield) { Yield("__str__", &PirShapeOrDataString); }));
  return axpr::MakeGlobalNaiveClassOps<symbol::ShapeOrDataDimExprs>(cls);
}

}  // namespace ap::paddle
