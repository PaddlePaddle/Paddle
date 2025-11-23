// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/pir/pass_method_class.h"

namespace ap::paddle {

struct PirPassMethodClass {
  using Self = ap::paddle::Pass;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self->pir_pass.get();
    std::ostringstream ss;
    ss << "<PirPass object at " << ptr << ">";
    return ss.str();
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetPirPassClass() {
  using Impl = PirPassMethodClass;
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "PirPass",
      [&](const auto& Yield) { Yield("__str__", &Impl::ToString); }));
  return axpr::MakeGlobalNaiveClassOps<ap::paddle::Pass>(cls);
}

}  // namespace ap::paddle
