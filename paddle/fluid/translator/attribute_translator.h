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

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/ir_context.h"

#pragma once

namespace paddle {
namespace translator {

class AttributeVisitor;

class AttributeTranslator {
 private:
  AttributeTranslator();
  AttributeVisitor* visitor;

 public:
  AttributeTranslator(const AttributeTranslator&) = delete;
  AttributeTranslator& operator=(const AttributeTranslator&) = delete;
  AttributeTranslator(AttributeTranslator&&) = delete;
  AttributeTranslator& operator=(AttributeTranslator&&) = delete;

  static auto& instance() {
    static AttributeTranslator attribute_translator;
    return attribute_translator;
  }

  ir::Attribute operator[](const framework::Attribute& attr);
};

}  // namespace translator
}  // namespace paddle
