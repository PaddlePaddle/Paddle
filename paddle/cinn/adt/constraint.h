// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <string>

#include "paddle/cinn/adt/m_expr.h"

namespace cinn::adt::constraint {

template <typename T>
class AllTemporal;

// AllTemporal [T] = [T]
template <typename T>
class AllTemporal<List<T>> final : public List<T> {
  using List<T>::List;
};

template <typename T>
class NoRedundentSpatial;

// NoRedundentSpatial [T] = [T]
template <typename T>
class NoRedundentSpatial<List<T>> final : public List<T> {
  using List<T>::List;
};

// Product T = [T]
template <typename T>
class Product final : public List<T> {
  using List<T>::List;
};

// Equal T = (T, T)
// Equal T0 T1 = (T0, T1)
template <typename T0, typename T1 = T0>
class Equal final : public Tuple<T0, T1> {
  using Tuple<T0, T1>::Tuple;
};

// UniqueName = std::string
using UniqueName = std::string;

// SymbolicDimSize = (UniqueName,)
using SymbolicDimSize = Tuple<UniqueName>;

// ConcreteDimSize = Int64
using ConcreteDimSize = int64_t;

// DimSize = SymbolicDimSize | ConcreteDimSize
using DimSize = Union<SymbolicDimSize, ConcreteDimSize>;

// SymbolicStride = Product SymbolicDimSize
using SymbolicStride = Product<SymbolicDimSize>;

// ConcreteStride = Int64
using ConcreteStride = int64_t;

// Stride = SymbolicStride | ConcreteStride
using Stride = Union<SymbolicStride, ConcreteStride>;

// SymbolicScheduleType = (UniqueName,)
using SymbolicScheduleType = Tuple<UniqueName>;

// ConcreteScheduleType = S0x | S0y | S0z | S1x | S1y | S1z | Temporal |
// Vectorize | Unroll
using ConcreteScheduleType = m_ir::ScheduleType;

// ScheduleType = SymbolicScheduleType | ConcreteScheduleType
using ScheduleType = Union<SymbolicScheduleType, ConcreteScheduleType>;

// SymbolicScheduleSize = (UniqueName,)
using SymbolicScheduleSize = Tuple<UniqueName>;

// ConcreteScheduleSize = Int64
using ConcreteScheduleSize = int64_t;

// ScheduleSize = SymbolicScheduleSize | ConcreteScheduleSize
using ScheduleSize = Union<SymbolicScheduleSize, ConcreteScheduleSize>;

// Equation = Equal DimSize
//          | Equal Stride
//          | Equal (Product DimSize)
//          | Equal ScheduleType
//          | Equal ScheduleSize
//          | Equal (Product DimSize) (Product ScheduleSize)
// clang-format off
using Equation = Union<Equal<DimSize>,
                       Equal<Stride>,
                       Equal<Product<DimSize>>,
                       Equal<ScheduleType>,
                       Equal<ScheduleSize>,
                       Equal<Product<DimSize>, Product<ScheduleSize>>>;
// clang-format on

// Constraint = Equation | AllTemporal [ScheduleType] | NoRedundentSpatial
// [ScheduleType]
using Constraint = Union<Equation,
                         AllTemporal<List<ScheduleType>>,
                         NoRedundentSpatial<List<ScheduleType>>>;

// Constraints = [Constraint]
using Constraints = List<Constraint>;

}  // namespace cinn::adt::constraint
