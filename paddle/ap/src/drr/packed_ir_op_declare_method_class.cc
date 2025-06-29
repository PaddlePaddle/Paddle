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

#include "paddle/ap/include/drr/packed_ir_op_declare_method_class.h"

namespace ap::drr {

struct SrcPtnPackedIrOpDeclareMethodClassImpl {
  using Self = drr::tSrcPtn<drr::PackedIrOpDeclare<drr::Node>>;
  using This = SrcPtnPackedIrOpDeclareMethodClassImpl;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }
};

struct ResPtnPackedIrOpDeclareMethodClassImpl {
  using Self = drr::tResPtn<drr::PackedIrOpDeclare<drr::Node>>;
  using This = ResPtnPackedIrOpDeclareMethodClassImpl;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnPackedIrOpDeclareClass() {
  using Impl = SrcPtnPackedIrOpDeclareMethodClassImpl;
  using TT = drr::Type<drr::tSrcPtn<drr::PackedIrOpDeclare<drr::Node>>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResPtnPackedIrOpDeclareClass() {
  using Impl = ResPtnPackedIrOpDeclareMethodClassImpl;
  using TT = drr::Type<drr::tResPtn<drr::PackedIrOpDeclare<drr::Node>>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
