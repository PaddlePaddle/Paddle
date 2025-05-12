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

#pragma once

#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace ap::dialect {

template <typename AttrT>
using ApPirAttributeImpl = std::variant<bool,
                                        int64_t,
                                        double,
                                        std::string,
                                        axpr::DataType,
                                        adt::List<AttrT>>;

struct ApPirAttribute : public ApPirAttributeImpl<ApPirAttribute> {
  using ApPirAttributeImpl<ApPirAttribute>::ApPirAttributeImpl;
  ADT_DEFINE_VARIANT_METHODS(ApPirAttributeImpl<ApPirAttribute>);

  static std::optional<ApPirAttribute> OptCastFrom(const axpr::Value& value) {
    const auto& ret = CastFrom(value);
    if (ret.HasError()) return std::nullopt;
    return ret.GetOkValue();
  }

  static adt::Result<ApPirAttribute> CastFrom(const axpr::Value& value) {
    using RetT = adt::Result<ApPirAttribute>;
    return value.Match(
        [](bool impl) -> RetT { return impl; },
        [](int64_t impl) -> RetT { return impl; },
        [](double impl) -> RetT { return impl; },
        [](const std::string& impl) -> RetT { return impl; },
        [](const axpr::DataType& impl) -> RetT { return impl; },
        [](const adt::List<axpr::Value>& lst) -> RetT {
          adt::List<ApPirAttribute> ret;
          ret->reserve(lst->size());
          for (const auto& elt_val : *lst) {
            ADT_LET_CONST_REF(ap_attr, ApPirAttribute::CastFrom(elt_val));
            ret->emplace_back(ap_attr);
          }
          return ret;
        },
        [](const auto&) -> RetT {
          return adt::errors::TypeError{
              "couldn't cast object from axpr::Value to ApPirAttribute"};
        });
  }

  adt::Result<pir::Attribute> CastToPirAttribute() const {
    return Match([&](const auto& impl) -> adt::Result<pir::Attribute> {
      return CastToPirAttributeImpl(impl);
    });
  }

  adt::Result<pir::Attribute> CastToPirAttributeImpl(bool impl) const {
    return pir::BoolAttribute::get(pir::IrContext::Instance(), impl);
  }

  adt::Result<pir::Attribute> CastToPirAttributeImpl(int64_t impl) const {
    return pir::Int64Attribute::get(pir::IrContext::Instance(), impl);
  }

  adt::Result<pir::Attribute> CastToPirAttributeImpl(double impl) const {
    return pir::DoubleAttribute::get(pir::IrContext::Instance(), impl);
  }

  adt::Result<pir::Attribute> CastToPirAttributeImpl(
      const std::string& impl) const {
    return pir::StrAttribute::get(pir::IrContext::Instance(), impl);
  }

  adt::Result<pir::Attribute> CastToPirAttributeImpl(
      const axpr::DataType& impl) const {
    ADT_LET_CONST_REF(phi_data_type, axpr::GetPhiDataTypeFromDataType(impl));
    return ::paddle::dialect::DataTypeAttribute::get(pir::IrContext::Instance(),
                                                     phi_data_type);
  }

  adt::Result<pir::Attribute> CastToPirAttributeImpl(
      const adt::List<ApPirAttribute>& impl) const {
    std::vector<pir::Attribute> vec;
    vec.resize(impl->size());
    for (const auto& ap_attr : *impl) {
      ADT_LET_CONST_REF(elt_attr, ap_attr.CastToPirAttribute());
      vec.emplace_back(elt_attr);
    }
    return pir::ArrayAttribute::get(pir::IrContext::Instance(), vec);
  }
};

}  // namespace ap::dialect
