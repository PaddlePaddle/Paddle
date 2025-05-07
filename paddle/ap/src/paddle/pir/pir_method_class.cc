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

#include "paddle/ap/include/paddle/pir/pir_method_class.h"
#include "paddle/ap/include/axpr/module_mgr.h"
#include "paddle/ap/include/paddle/pir_node.h"

namespace ap::paddle {

void ForceLinkPir() {
  // Do nothing.
}

template <typename Builder>
void DefineMethods(Builder* m) {
  m->Def("UndefinedPlace", &CreateUndefinedPlace);
  m->Def("CPUPlace", &CreateCPUPlace);
  m->Def("GPUPlace", &CreateGPUPlace);
  m->Def("GPUPinnedPlace", &CreateGPUPinnedPlace);
  m->Def("XPUPlace", &CreateXPUPlace);
  m->Def("IPUPlace", &CreateIPUPlace);
  m->Def("CustomPlace", &CreateCustomPlace);
#define DEF_MAKE_ATTRIBUTE(attr_type) \
  m->Def(attr_type::name(), &MakePirAttributeImpl<attr_type>::Call);
  FOR_EACH_PIR_ATTRIBUTE_TYPE(DEF_MAKE_ATTRIBUTE);
#undef DEF_MAKE_ATTRIBUTE

#define DEF_MAKE_TYPE(cls) m->Def(cls::name(), &MakePirTypeImpl<cls>::Call);
  FOR_EACH_PIR_ALTERNATIVE_TYPE(DEF_MAKE_TYPE);
#undef DEF_MAKE_TYPE
}

REGISTER_AP_BUILTIN_MODULE("pir", [](auto* m) { DefineMethods(m); });

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetPirClass() {
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>("pir", [&](const auto& Yield) {
        Yield("UndefinedPlace", &CreateUndefinedPlace);
        Yield("CPUPlace", &CreateCPUPlace);
        Yield("GPUPlace", &CreateGPUPlace);
        Yield("GPUPinnedPlace", &CreateGPUPinnedPlace);
        Yield("XPUPlace", &CreateXPUPlace);
        Yield("IPUPlace", &CreateIPUPlace);
        Yield("CustomPlace", &CreateCustomPlace);
#define YIELD_MAKE_ATTRIBUTE(attr_type) \
  Yield(attr_type::name(), &MakePirAttributeImpl<attr_type>::Call);
        FOR_EACH_PIR_ATTRIBUTE_TYPE(YIELD_MAKE_ATTRIBUTE);
#undef YIELD_MAKE_ATTRIBUTE

#define YIELD_MAKE_TYPE(cls) Yield(cls::name(), &MakePirTypeImpl<cls>::Call);
        FOR_EACH_PIR_ALTERNATIVE_TYPE(YIELD_MAKE_TYPE);
#undef YIELD_MAKE_TYPE
      }));
  return axpr::MakeGlobalNaiveClassOps<Pir>(cls);
}

}  // namespace ap::paddle
