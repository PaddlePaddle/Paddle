// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/tags.h"
#include "paddle/cinn/adt/unique_id.h"

namespace cinn::adt {

// Dim = tDim UniqueId
using Dim = tDim<UniqueId>;
// DimTuple = [Dim]
using DimTuple = List<Dim>;

DEFINE_ADT_UNION(Constant, std::int64_t, Dim, List<Constant>);

OVERLOAD_OPERATOR_EQ_NE(Constant, UnionEqual);

// EquationStaticValue = Dim | std::int64_t
DEFINE_ADT_UNION(EquationStaticValue, Dim, std::int64_t);
OVERLOAD_OPERATOR_EQ_NE(EquationStaticValue, UnionEqual);

using EquationStaticLogical = Logical<EquationStaticValue>;

inline std::size_t GetHashValue(const Constant& c);

inline std::size_t GetHashValueImpl(const std::int64_t& c) { return c; }
inline std::size_t GetHashValueImpl(const Dim& c) {
  return c.value().unique_id();
}
inline std::size_t GetHashValueImpl(const List<Constant>& c) {
  std::size_t ret = 0;
  for (const auto& c_item : *c) {
    ret = hash_combine(ret, GetHashValue(c_item));
  }
  return ret;
}

OVERRIDE_UNION_GET_HASH_VALUE(Constant);

}  // namespace cinn::adt

namespace std {

template <>
struct hash<::cinn::adt::Dim> final {
  std::size_t operator()(const ::cinn::adt::Dim& dim) const {
    return dim.value().unique_id();
  }
};

template <>
struct hash<cinn::adt::Constant> {
  std::size_t operator()(const cinn::adt::Constant& constant) const {
    return GetHashValue(constant);
  }
};

}  // namespace std
