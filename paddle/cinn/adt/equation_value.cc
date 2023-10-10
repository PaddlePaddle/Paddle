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

#include <string>

#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/schedule_descriptor.h"

namespace cinn::adt {

std::string DebugStringImpl(std::int64_t c) { return std::to_string(c); }
std::string DebugStringImpl(const tStride<UniqueId>& c) {
  return std::string("stride_") + std::to_string(c.value().unique_id());
}

std::string DebugStringImpl(const tDim<UniqueId>& c) {
  return std::string("dim_") + std::to_string(c.value().unique_id());
}

std::string DebugStringImpl(const List<Constant>& c) {
  std::string ret{"["};
  std::size_t count = 0;
  for (const auto& tmp : *c) {
    if (count++ > 0) {
      ret += ", ";
    }
    ret += DebugString(tmp);
  }
  ret += "]";
  return ret;
}

std::string DebugStringImpl(const Neg<Constant>& c) {
  return std::string("-") + DebugString(std::get<0>(c.tuple()));
}

std::string DebugStringImpl(const Add<Constant, Constant>& c) {
  return DebugString(std::get<0>(c.tuple())) + " + " +
         DebugString(std::get<1>(c.tuple()));
}

std::string DebugStringImpl(const Mul<Constant, Constant>& c) {
  return DebugString(std::get<0>(c.tuple())) + " * " +
         DebugString(std::get<1>(c.tuple()));
}

std::string DebugString(const Constant& c) {
  return std::visit([&](const auto& impl) { return DebugStringImpl(impl); },
                    c.variant());
}

std::string DebugStringImpl(const Undefined&) { return "Undefined"; }
std::string DebugStringImpl(const Ok&) { return "Ok"; }

std::string DebugStringImpl(const Iterator& iterator) {
  return std::string("i_") + std::to_string(iterator.value().unique_id());
}

std::string DebugStringImpl(const Constant& c) { return DebugString(c); }

std::string DebugStringImpl(const List<Value>& values) {
  std::string ret = "[";
  std::size_t count = 0;
  for (const auto& value : *values) {
    if (count++ > 0) {
      ret += ", ";
    }
    ret += DebugString(value);
  }
  ret += "]";
  return ret;
}

std::string DebugStringImpl(const IndexDotValue<Value, Constant>& index_dot) {
  const auto& [iters, constant] = index_dot.tuple();
  return std::string() + "IndexDotValue(" + DebugString(iters) + ", " +
         DebugString(constant) + ")";
}

std::string DebugStringImpl(
    const IndexUnDotValue<Value, Constant>& index_undot) {
  const auto& [index, constant] = index_undot.tuple();
  return std::string() + "IndexUnDotValue(" + DebugString(index) + ", " +
         DebugString(constant) + ")";
}

std::string DebugStringImpl(const ConstantAdd<Value>& constant_add) {
  const auto& [value, constant] = constant_add.tuple();
  return std::string() + "ConstantAdd(" + DebugString(value) + ", " +
         DebugString(constant) + ")";
}

std::string DebugStringImpl(const ConstantDiv<Value>& constant_div) {
  const auto& [value, constant] = constant_div.tuple();
  return std::string() + "ConstantDiv(" + DebugString(value) + ", " +
         DebugString(constant) + ")";
}

std::string DebugStringImpl(const ConstantMod<Value>& constant_mod) {
  const auto& [value, constant] = constant_mod.tuple();
  return std::string() + "ConstantDiv(" + DebugString(value) + ", " +
         DebugString(constant) + ")";
}

std::string DebugStringImpl(const ListGetItem<Value, Constant>& list_get_item) {
  const auto& [value, constant] = list_get_item.tuple();
  return std::string() + "ListGetItem(" + DebugString(value) + ", " +
         DebugString(constant) + ")";
}

std::string DebugStringImpl(const PtrGetItem<Value>& ptr_get_item) {
  const auto& [unique_id, value] = ptr_get_item.tuple();
  return std::string() + "PtrGetItem(" +
         std::to_string(unique_id.value().unique_id()) + ", " +
         DebugString(value) + ")";
}

std::string DebugString(const Value& value) {
  return std::visit([&](const auto& impl) { return DebugStringImpl(impl); },
                    value.variant());
}

}  // namespace cinn::adt
