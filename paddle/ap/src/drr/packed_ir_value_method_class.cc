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

#include "paddle/ap/include/drr/packed_ir_value_method_class.h"

namespace ap::drr {

struct SrcPtnPackedIrValueMethodClassImpl {
  using Self = drr::tSrcPtn<drr::PackedIrValue<drr::Node>>;
  using This = SrcPtnPackedIrValueMethodClassImpl;

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

  static adt::Result<axpr::Value> Starred(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    return axpr::Starred<axpr::Value>{adt::List<axpr::Value>{self_val}};
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnPackedIrValueClass() {
  using Impl = SrcPtnPackedIrValueMethodClassImpl;
  using TT = drr::Type<drr::tSrcPtn<drr::PackedIrValue<drr::Node>>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

struct StarredSrcPtnPackedIrValueMethodClassImpl {
  using Self = drr::tStarred<drr::tSrcPtn<drr::PackedIrValue<axpr::Value>>>;
  using This = StarredSrcPtnPackedIrValueMethodClassImpl;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.value().value().__adt_rc_shared_ptr_raw_ptr();
    std::ostringstream ss;
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.value().value().__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetStarredSrcPtnPackedIrValueClass() {
  using Impl = StarredSrcPtnPackedIrValueMethodClassImpl;
  using TT =
      drr::Type<drr::tStarred<drr::tSrcPtn<drr::PackedIrValue<drr::Node>>>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

struct ResPtnPackedIrValueMethodClassImpl {
  using Self = drr::tResPtn<drr::PackedIrValue<drr::Node>>;
  using This = ResPtnPackedIrValueMethodClassImpl;

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

  static adt::Result<axpr::Value> Starred(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    return axpr::Starred<axpr::Value>{adt::List<axpr::Value>{self_val}};
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResPtnPackedIrValueClass() {
  using Impl = ResPtnPackedIrValueMethodClassImpl;
  using TT = drr::Type<drr::tResPtn<drr::PackedIrValue<drr::Node>>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

struct StarredResPtnPackedIrValueMethodClassImpl {
  using This = StarredResPtnPackedIrValueMethodClassImpl;
  using Self = drr::tStarred<drr::tResPtn<drr::PackedIrValue<drr::Node>>>;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const void* ptr = self.value().value().__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.value().value().__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetStarredResPtnPackedIrValueClass() {
  using Impl = StarredResPtnPackedIrValueMethodClassImpl;
  using TT =
      drr::Type<drr::tStarred<drr::tResPtn<drr::PackedIrValue<drr::Node>>>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
