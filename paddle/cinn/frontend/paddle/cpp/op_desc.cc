// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/paddle/cpp/op_desc.h"

#include <cstdint>
#include <utility>

namespace cinn::frontend::paddle::cpp {

inline std::string AttrTypeToString(
    paddle::cpp::OpDescAPI::AttrType attr_type) {
  using AttrType = paddle::cpp::OpDescAPI::AttrType;
  switch (attr_type) {
#define EXPAND_SWITCH_CASE(ATTR_TYPE) \
  case AttrType::ATTR_TYPE:           \
    return #ATTR_TYPE;
    EXPAND_SWITCH_CASE(INT)
    EXPAND_SWITCH_CASE(FLOAT)
    EXPAND_SWITCH_CASE(STRING)
    EXPAND_SWITCH_CASE(INTS)
    EXPAND_SWITCH_CASE(FLOATS)
    EXPAND_SWITCH_CASE(STRINGS)
    EXPAND_SWITCH_CASE(BOOLEAN)
    EXPAND_SWITCH_CASE(LONG)
    EXPAND_SWITCH_CASE(LONGS)
    EXPAND_SWITCH_CASE(FLOAT64S)
    EXPAND_SWITCH_CASE(FLOAT64)
    EXPAND_SWITCH_CASE(SCALAR)
    EXPAND_SWITCH_CASE(SCALARS)
#undef EXPAND_SWITCH_CASE
  }
  return "Invlid AttrType";
}

#define SET_ATTR_IMPL(T, repr__)                                 \
  template <>                                                    \
  void OpDesc::SetAttr<T>(const std::string& name, const T& v) { \
    attr_types_[name] = AttrType::repr__;                        \
    attrs_[name] = v;                                            \
  }

SET_ATTR_IMPL(int32_t, INT);
SET_ATTR_IMPL(float, FLOAT);
SET_ATTR_IMPL(double, FLOAT64);
SET_ATTR_IMPL(std::string, STRING);
SET_ATTR_IMPL(bool, BOOLEAN);
SET_ATTR_IMPL(int64_t, LONG);
SET_ATTR_IMPL(std::vector<int>, INTS);
SET_ATTR_IMPL(std::vector<float>, FLOATS);
SET_ATTR_IMPL(std::vector<double>, FLOAT64S);
SET_ATTR_IMPL(std::vector<std::string>, STRINGS);
SET_ATTR_IMPL(std::vector<bool>, BOOLEANS);
SET_ATTR_IMPL(std::vector<int64_t>, LONGS);

#undef SET_ATTR_IMPL

std::pair<OpDesc::attrs_t::const_iterator, OpDesc::attr_types_t::const_iterator>
FindAttr(const OpDesc& desc, const std::string& name) {
  auto it = desc.attrs().find(name);
  CHECK(it != desc.attrs().end())
      << "No attributes called " << name << " found";
  auto attr_it = desc.attr_types().find(name);
  CHECK(attr_it != desc.attr_types().end());
  return std::make_pair(it, attr_it);
}

#define GET_IMPL_ONE(T, repr__)                                              \
  template <>                                                                \
  T OpDesc::GetAttr<T>(const std::string& name) const {                      \
    auto pair = FindAttr(*this, name);                                       \
    CHECK(pair.second->second == AttrType::repr__)                           \
        << "The op \"" << Type() << "\"'s attrbute \"" << pair.second->first \
        << "\"'s type doesn't match the target type! Try get \"" << #repr__  \
        << "\", but real \"" << AttrTypeToString(pair.second->second)        \
        << "\". Please check.";                                              \
    return absl::any_cast<T>(pair.first->second);                            \
  }

GET_IMPL_ONE(int32_t, INT);
GET_IMPL_ONE(float, FLOAT);
GET_IMPL_ONE(double, FLOAT64);
GET_IMPL_ONE(std::string, STRING);
GET_IMPL_ONE(bool, BOOLEAN);
GET_IMPL_ONE(int64_t, LONG);
GET_IMPL_ONE(std::vector<int>, INTS);
GET_IMPL_ONE(std::vector<float>, FLOATS);
GET_IMPL_ONE(std::vector<double>, FLOAT64S);
GET_IMPL_ONE(std::vector<std::string>, STRINGS);
GET_IMPL_ONE(std::vector<bool>, BOOLEANS);
GET_IMPL_ONE(std::vector<int64_t>, LONGS);

#undef GET_IMPL_ONE

std::vector<std::string> OpDesc::OutputArgumentNames() const {
  std::vector<std::string> res;
  for (const auto& x : outputs_) res.push_back(x.first);
  return res;
}

std::vector<std::string> OpDesc::input_vars() const {
  std::vector<std::string> res;
  for (const auto& arg : InputArgumentNames()) {
    for (auto& vars : Input(arg)) {
      res.emplace_back(vars.begin(), vars.end());
    }
  }
  return res;
}

std::vector<std::string> OpDesc::output_vars() const {
  std::vector<std::string> res;
  for (const auto& arg : OutputArgumentNames()) {
    for (auto& vars : Output(arg)) {
      res.emplace_back(vars.begin(), vars.end());
    }
  }
  return res;
}

std::vector<std::string> OpDesc::InputArgumentNames() const {
  std::vector<std::string> res;
  for (const auto& x : inputs_) res.push_back(x.first);
  return res;
}

std::vector<std::string> OpDesc::Input(const std::string& param) const {
  auto it = inputs_.find(param);
  CHECK(it != inputs_.end());
  return it->second;
}

std::vector<std::string> OpDesc::Output(const std::string& param) const {
  auto it = outputs_.find(param);
  CHECK(it != outputs_.end());
  return it->second;
}

bool OpDesc::HasOutput(const std::string& param) const {
  auto it = outputs_.find(param);
  return it != outputs_.end();
}

}  // namespace cinn::frontend::paddle::cpp
