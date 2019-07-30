// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/model_parser/cpp/op_desc.h"
#include <set>
#include <utility>

namespace paddle {
namespace lite {
namespace cpp {

#define SET_ATTR_IMPL(T, repr__)                                 \
  template <>                                                    \
  void OpDesc::SetAttr<T>(const std::string& name, const T& v) { \
    attr_types_[name] = AttrType::repr__;                        \
    attrs_[name].set<T>(v);                                      \
  }

SET_ATTR_IMPL(int32_t, INT);
SET_ATTR_IMPL(int64_t, LONG);
SET_ATTR_IMPL(float, FLOAT);
SET_ATTR_IMPL(std::string, STRING);
SET_ATTR_IMPL(bool, BOOLEAN);
SET_ATTR_IMPL(std::vector<int>, INTS);
SET_ATTR_IMPL(std::vector<float>, FLOATS);
SET_ATTR_IMPL(std::vector<std::string>, STRINGS);
SET_ATTR_IMPL(std::vector<int64_t>, LONGS);

std::pair<OpDesc::attrs_t::const_iterator, OpDesc::attr_types_t::const_iterator>
FindAttr(const cpp::OpDesc& desc, const std::string& name) {
  auto it = desc.attrs().find(name);
  CHECK(it != desc.attrs().end()) << "No attributes called " << name
                                  << " found";
  auto attr_it = desc.attr_types().find(name);
  CHECK(attr_it != desc.attr_types().end());
  return std::make_pair(it, attr_it);
}

#define GET_IMPL_ONE(T, repr__)                                          \
  template <>                                                            \
  T OpDesc::GetAttr<T>(const std::string& name) const {                  \
    auto pair = FindAttr(*this, name);                                   \
    CHECK(pair.second->second == AttrType::repr__)                       \
        << "required type is " << #repr__ << " not match the true type"; \
    return pair.first->second.get<T>();                                  \
  }

GET_IMPL_ONE(int32_t, INT);
GET_IMPL_ONE(int64_t, LONG);
GET_IMPL_ONE(float, FLOAT);
GET_IMPL_ONE(std::string, STRING);
GET_IMPL_ONE(bool, BOOLEAN);
GET_IMPL_ONE(std::vector<int64_t>, LONGS);
GET_IMPL_ONE(std::vector<float>, FLOATS);
GET_IMPL_ONE(std::vector<int>, INTS);
GET_IMPL_ONE(std::vector<std::string>, STRINGS);

}  // namespace cpp
}  // namespace lite
}  // namespace paddle
