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

namespace paddle {
namespace lite {
namespace cpp {

template <>
void OpDesc::SetAttr<int32_t>(const std::string& name, const int32_t& v) {
  attr_types_[name] = AttrType::INT;
  attrs_[name].set<int>(v);
}
template <>
void OpDesc::SetAttr<float>(const std::string& name, const float& v) {
  attr_types_[name] = AttrType::FLOAT;
  attrs_[name].set<float>(v);
}

template <>
int32_t OpDesc::GetAttr<int32_t>(const std::string& name) const {
  auto it = attrs_.find(name);
  CHECK(it != attrs_.end()) << "No attributes called " << name << " found";
  auto attr_it = attr_types_.find(name);
  CHECK(attr_it != attr_types_.end());
  CHECK(attr_it->second == AttrType::INT);
  return it->second.get<int32_t>();
}

template <>
float OpDesc::GetAttr<float>(const std::string& name) const {
  auto it = attrs_.find(name);
  CHECK(it != attrs_.end()) << "No attributes called " << name << " found";
  auto attr_it = attr_types_.find(name);
  CHECK(attr_it != attr_types_.end());
  CHECK(attr_it->second == AttrType::INT);
  return it->second.get<float>();
}

}  // namespace cpp
}  // namespace lite
}  // namespace paddle
