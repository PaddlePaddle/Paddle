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

#pragma once

#include "paddle/common/adt_type_id.h"

namespace pir {

class Attribute;
class BoolAttribute;
class Complex64Attribute;
class Complex128Attribute;
class FloatAttribute;
class DoubleAttribute;
class Int32Attribute;
class IndexAttribute;
class Int64Attribute;
class PointerAttribute;
class TypeAttribute;
class StrAttribute;
class ArrayAttribute;
class TensorNameAttribute;

}  // namespace pir

namespace pir::shape {

class SymbolAttribute;

}

namespace paddle::dialect {

class KernelAttribute;
class IntArrayAttribute;
class ScalarAttribute;
class DataTypeAttribute;
class PlaceAttribute;
class DataLayoutAttribute;

}  // namespace paddle::dialect

namespace cinn::dialect {

class GroupInfoAttribute;
class CINNKernelInfoAttribute;
class FusionTrackerPtrAttribute;

}  // namespace cinn::dialect

namespace cinn::dialect::ir {

class UnclassifiedAttribute {};

// clang-format off
#define FOR_EACH_PIR_ATTRIBUTE_TYPE(__macro)        \
  __macro(pir::BoolAttribute)                       \
  __macro(pir::Complex64Attribute)                  \
  __macro(pir::Complex128Attribute)                 \
  __macro(pir::FloatAttribute)                      \
  __macro(pir::DoubleAttribute)                     \
  __macro(pir::Int32Attribute)                      \
  __macro(pir::IndexAttribute)                      \
  __macro(pir::Int64Attribute)                      \
  __macro(pir::PointerAttribute)                    \
  __macro(pir::TypeAttribute)                       \
  __macro(pir::StrAttribute)                        \
  __macro(pir::ArrayAttribute)                      \
  __macro(pir::TensorNameAttribute)                 \
  __macro(pir::shape::SymbolAttribute)              \
  __macro(paddle::dialect::KernelAttribute)         \
  __macro(paddle::dialect::IntArrayAttribute)       \
  __macro(paddle::dialect::ScalarAttribute)         \
  __macro(paddle::dialect::DataTypeAttribute)       \
  __macro(paddle::dialect::PlaceAttribute)          \
  __macro(paddle::dialect::DataLayoutAttribute)     \
  __macro(cinn::dialect::GroupInfoAttribute)        \
  __macro(cinn::dialect::CINNKernelInfoAttribute)   \
  __macro(cinn::dialect::FusionTrackerPtrAttribute)
// clang-format on

using AttrAdtTypeIdBase = ::common::AdtBaseTypeId<
#define AS_ATTR_ADT_TYPE_ID_ALTERNATIVE(cls) cls,
    FOR_EACH_PIR_ATTRIBUTE_TYPE(AS_ATTR_ADT_TYPE_ID_ALTERNATIVE)
#undef AS_ATTR_ADT_TYPE_ID_ALTERNATIVE
        UnclassifiedAttribute>;

struct AttrAdtTypeId : public AttrAdtTypeIdBase {
  using AttrAdtTypeIdBase::AttrAdtTypeIdBase;
};

AttrAdtTypeId GetAttrAdtTypeId(const pir::Attribute& attribute);

}  // namespace cinn::dialect::ir
