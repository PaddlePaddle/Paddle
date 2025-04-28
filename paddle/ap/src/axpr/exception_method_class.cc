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

#include "paddle/ap/include/axpr/exception_method_class.h"

namespace ap::axpr {

struct ExceptionMethodClass {
  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Exception>());
    std::ostringstream ss;
    ss << self.value().class_name() << ": " << self.value().msg();
    return ss.str();
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetExceptionClass() {
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("Exception", [&](const auto& Yield) {
        Yield("__str__", &ExceptionMethodClass::ToString);
      }));
  return axpr::MakeGlobalNaiveClassOps<Exception>(cls);
}

}  // namespace ap::axpr
