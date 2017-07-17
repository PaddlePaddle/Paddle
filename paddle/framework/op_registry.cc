/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <paddle/framework/op_registry.h>

namespace paddle {
namespace framework {

template <>
void AttrTypeHelper::SetAttrType<int>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::INT);
}

template <>
void AttrTypeHelper::SetAttrType<float>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::FLOAT);
}

template <>
void AttrTypeHelper::SetAttrType<std::string>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::STRING);
}

template <>
void AttrTypeHelper::SetAttrType<std::vector<int>>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::INTS);
}

template <>
void AttrTypeHelper::SetAttrType<std::vector<float>>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::FLOATS);
}

template <>
void AttrTypeHelper::SetAttrType<std::vector<std::string>>(AttrProto* attr) {
  attr->set_type(paddle::framework::AttrType::STRINGS);
}
}  // namespace framework
}  // namespace paddle
