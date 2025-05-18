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

#include "paddle/ap/include/drr/unbound_ir_value_method_class.h"

namespace ap::drr {

struct UnboundIrValueMethodClassImpl {
  using This = UnboundIrValueMethodClassImpl;
  using Self = UnboundIrValue<drr::Node>;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.__adt_rc_shared_ptr_raw_ptr();
    std::ostringstream ss;
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }

  static adt::Result<axpr::Value> SetAttr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(attr_name, args.at(0).template CastTo<std::string>());
    const auto& attr_val = args.at(1);
    if (attr_name == "type") {
      ADT_RETURN_IF_ERR(OpTensorPatternCtxHelper{}.SetType(self, attr_val));
      return adt::Nothing{};
    } else {
      return adt::errors::AttributeError{
          std::string(axpr::GetTypeName(self_val)) + " '" + self->name +
          "' has no attribute '" + attr_name + "'"};
    }
  }

  static adt::Result<axpr::Value> Starred(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    UnboundPackedIrValue<drr::Node> packed_ir_value{self->name,
                                                    self->tensor_pattern_ctx};
    DrrValueHelper helper{};
    axpr::Value starred{helper.CastToAxprValue(packed_ir_value)};
    return axpr::Starred<axpr::Value>{adt::List<axpr::Value>{starred}};
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetUnboundIrValueClass() {
  using Impl = UnboundIrValueMethodClassImpl;
  using TT = drr::Type<UnboundIrValue<drr::Node>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__starred__", &Impl::Starred);
        Define("__setattr__", &Impl::SetAttr);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
