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

#include "paddle/framework/attribute.h"

#include <vector>

namespace paddle {
namespace framework {

template <>
AttrType AttrTypeID<int>() {
  return INT;
}
template <>
AttrType AttrTypeID<float>() {
  return FLOAT;
}
template <>
AttrType AttrTypeID<std::string>() {
  return STRING;
}
template <>
AttrType AttrTypeID<std::vector<int>>() {
  return INTS;
}
template <>
AttrType AttrTypeID<std::vector<float>>() {
  return FLOATS;
}
template <>
AttrType AttrTypeID<std::vector<std::string>>() {
  return STRINGS;
}

Attribute GetAttrValue(const AttrDesc& attr_desc) {
  switch (attr_desc.type()) {
    case paddle::framework::AttrType::INT: {
      return attr_desc.i();
    }
    case paddle::framework::AttrType::FLOAT: {
      return attr_desc.f();
    }
    case paddle::framework::AttrType::STRING: {
      return attr_desc.s();
    }
    case paddle::framework::AttrType::INTS: {
      std::vector<int> val(attr_desc.ints_size());
      for (int i = 0; i < attr_desc.ints_size(); ++i) {
        val[i] = attr_desc.ints(i);
      }
      return val;
    }
    case paddle::framework::AttrType::FLOATS: {
      std::vector<float> val(attr_desc.floats_size());
      for (int i = 0; i < attr_desc.floats_size(); ++i) {
        val[i] = attr_desc.floats(i);
      }
      return val;
    }
    case paddle::framework::AttrType::STRINGS: {
      std::vector<std::string> val(attr_desc.strings_size());
      for (int i = 0; i < attr_desc.strings_size(); ++i) {
        val[i] = attr_desc.strings(i);
      }
      return val;
    }
  }
  PADDLE_ENFORCE(false, "Unknown OpDesc::AttrDesc::type !");
  return boost::blank();
}

}  // namespace framework
}  // namespace paddle
