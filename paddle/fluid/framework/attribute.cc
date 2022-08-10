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
#include "paddle/utils/blank.h"

namespace paddle {
namespace framework {

paddle::any GetAttrValue(const Attribute& attr) {
  switch (AttrTypeID(attr)) {
    case proto::AttrType::INT:
      return PADDLE_GET_CONST(int, attr);
    case proto::AttrType::FLOAT:
      return PADDLE_GET_CONST(float, attr);
    case proto::AttrType::STRING:
      return PADDLE_GET_CONST(std::string, attr);
    case proto::AttrType::INTS:
      return PADDLE_GET_CONST(std::vector<int>, attr);
    case proto::AttrType::FLOATS:
      return PADDLE_GET_CONST(std::vector<float>, attr);
    case proto::AttrType::STRINGS:
      return PADDLE_GET_CONST(std::vector<std::string>, attr);
    case proto::AttrType::BOOLEAN:
      return PADDLE_GET_CONST(bool, attr);
    case proto::AttrType::BOOLEANS:
      return PADDLE_GET_CONST(std::vector<bool>, attr);
    case proto::AttrType::LONG:
      return PADDLE_GET_CONST(int64_t, attr);
    case proto::AttrType::LONGS:
      return PADDLE_GET_CONST(std::vector<int64_t>, attr);
    case proto::AttrType::FLOAT64S:
      return PADDLE_GET_CONST(std::vector<double>, attr);
    case proto::AttrType::VAR:
      return PADDLE_GET_CONST(VarDesc*, attr);
    case proto::AttrType::VARS:
      return PADDLE_GET_CONST(std::vector<VarDesc*>, attr);
    case proto::AttrType::BLOCK:
      return PADDLE_GET_CONST(BlockDesc*, attr);
    case proto::AttrType::BLOCKS:
      return PADDLE_GET_CONST(std::vector<BlockDesc*>, attr);
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported Attribute value type `%s` for phi.",
          platform::demangle(attr.type().name())));
  }
}

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

    case proto::AttrType::FLOAT64S: {
      std::vector<double> val(attr_desc.float64s_size());
      for (int i = 0; i < attr_desc.float64s_size(); ++i) {
        val[i] = attr_desc.float64s(i);
      }
      return val;
    }

    default:
      PADDLE_THROW(platform::errors::Unavailable("Unsupport attribute type %d.",
                                                 attr_desc.type()));
  }
  return paddle::blank();
}

}  // namespace framework
}  // namespace paddle
