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

#include "paddle/fluid/translator/attribute_translator.h"

#include <string>
#include <vector>

#include "paddle/utils/variant.h"

namespace paddle {
namespace translator {

class AttributeVisitor {
 public:
  ir::IrContext* ctx;
  AttributeVisitor() { ctx = ir::IrContext::Instance(); }
  ~AttributeVisitor() {}

 public:
  ir::Attribute operator()(int i) { return ir::Int64_tAttribute::get(ctx, i); }

  ir::Attribute operator()(float f) { return ir::FloatAttribute::get(ctx, f); }

  ir::Attribute operator()(bool b) { return ir::BoolAttribute::get(ctx, b); }

  ir::Attribute operator()(double d) {
    return ir::DoubleAttribute::get(ctx, d);
  }

  ir::Attribute operator()(std::string str) {
    return ir::StrAttribute::get(ctx, str);
  }

  template <typename T>
  ir::Attribute operator()(T attr) {
    return ir::Attribute(nullptr);
  }
};

AttributeTranslator::AttributeTranslator() { visitor = new AttributeVisitor(); }

ir::Attribute AttributeTranslator::operator[](
    const framework::Attribute& attr) {
  return paddle::visit(*visitor, attr);
}

}  // namespace translator
}  // namespace paddle
