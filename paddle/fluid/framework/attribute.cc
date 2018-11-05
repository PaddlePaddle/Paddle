/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/attribute.h"

#include <vector>

namespace paddle {
namespace framework {

Attribute GetAttrValue(const proto::OpDesc::Attr& attr_desc) {
  switch (attr_desc.type()) {
    case proto::AttrType::BOOLEAN: {
      return attr_desc.b();
    }
    case proto::AttrType::INT: {
      return attr_desc.i();
    }
    case proto::AttrType::FLOAT: {
      return attr_desc.f();
    }
    case proto::AttrType::STRING: {
      return attr_desc.s();
    }
    case proto::AttrType::BOOLEANS: {
      std::vector<bool> val(attr_desc.bools_size());
      for (int i = 0; i < attr_desc.bools_size(); ++i) {
        val[i] = attr_desc.bools(i);
      }
      return val;
    }
    case proto::AttrType::INTS: {
      std::vector<int> val(attr_desc.ints_size());
      for (int i = 0; i < attr_desc.ints_size(); ++i) {
        val[i] = attr_desc.ints(i);
      }
      return val;
    }
    case proto::AttrType::FLOATS: {
      std::vector<float> val(attr_desc.floats_size());
      for (int i = 0; i < attr_desc.floats_size(); ++i) {
        val[i] = attr_desc.floats(i);
      }
      return val;
    }
    case proto::AttrType::STRINGS: {
      std::vector<std::string> val(attr_desc.strings_size());
      for (int i = 0; i < attr_desc.strings_size(); ++i) {
        val[i] = attr_desc.strings(i);
      }
      return val;
    }
    case proto::AttrType::LONG: {
      return attr_desc.l();
    }
    case proto::AttrType::LONGS: {
      std::vector<int64_t> val(attr_desc.longs_size());
      for (int i = 0; i < attr_desc.longs_size(); ++i) {
        val[i] = attr_desc.longs(i);
      }
      return val;
    }
    default:
      PADDLE_THROW("Unsupport attr type %d", attr_desc.type());
  }
  return boost::blank();
}

}  // namespace framework
}  // namespace paddle
