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

static ProgramDesc* g_program_desc = nullptr;

ProgramDesc& GetProgramDesc() {
  if (g_program_desc == nullptr) {
    g_program_desc = new ProgramDesc();
    auto root_block = g_program_desc->mutable_blocks()->Add();
    root_block->set_idx(0);
    root_block->set_parent_idx(-1);
  }
  return *g_program_desc;
}

Attribute GetAttrValue(const OpDesc::Attr& attr_desc) {
  switch (attr_desc.type()) {
    case framework::AttrType::BOOLEAN: {
      return attr_desc.b();
    }
    case framework::AttrType::INT: {
      return attr_desc.i();
    }
    case framework::AttrType::FLOAT: {
      return attr_desc.f();
    }
    case framework::AttrType::STRING: {
      return attr_desc.s();
    }
    case framework::AttrType::BOOLEANS: {
      std::vector<bool> val(attr_desc.bools_size());
      for (int i = 0; i < attr_desc.bools_size(); ++i) {
        val[i] = attr_desc.bools(i);
      }
      return val;
    }
    case framework::AttrType::INTS: {
      std::vector<int> val(attr_desc.ints_size());
      for (int i = 0; i < attr_desc.ints_size(); ++i) {
        val[i] = attr_desc.ints(i);
      }
      return val;
    }
    case framework::AttrType::FLOATS: {
      std::vector<float> val(attr_desc.floats_size());
      for (int i = 0; i < attr_desc.floats_size(); ++i) {
        val[i] = attr_desc.floats(i);
      }
      return val;
    }
    case framework::AttrType::STRINGS: {
      std::vector<std::string> val(attr_desc.strings_size());
      for (int i = 0; i < attr_desc.strings_size(); ++i) {
        val[i] = attr_desc.strings(i);
      }
      return val;
    }
    case framework::AttrType::BLOCK: {
      return GetProgramDesc().mutable_blocks(attr_desc.block_idx());
    }
  }
  PADDLE_ENFORCE(false, "Unknown OpDesc::AttrDesc::type !");
  return boost::blank();
}

}  // namespace framework
}  // namespace paddle
