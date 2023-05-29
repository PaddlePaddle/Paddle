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

#include "paddle/fluid/dialect/pd_attribute.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/variant.h"

namespace paddle {
namespace translator {

class AttributeVisitor {
 public:
  ir::IrContext* ctx;
  AttributeVisitor() { ctx = ir::IrContext::Instance(); }
  ~AttributeVisitor() {}

 public:
  ir::Attribute operator()(int i) { return ir::Int32_tAttribute::get(ctx, i); }

  ir::Attribute operator()(float f) { return ir::FloatAttribute::get(ctx, f); }

  ir::Attribute operator()(bool b) { return ir::BoolAttribute::get(ctx, b); }

  ir::Attribute operator()(double d) {
    return ir::DoubleAttribute::get(ctx, d);
  }

  ir::Attribute operator()(std::string str) {
    return ir::StrAttribute::get(ctx, str);
  }

  ir::Attribute operator()(const paddle::experimental::Scalar& scalar) {
    return paddle::dialect::ScalarAttribute::get(ctx, scalar);
  }

  ir::Attribute operator()(const std::vector<std::string>& strs) {
    std::vector<ir::Attribute> attrs;
    attrs.reserve(strs.size());
    for (const auto& v : strs) {
      attrs.push_back(ir::StrAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  ir::Attribute operator()(const std::vector<float>& fs) {
    std::vector<ir::Attribute> attrs;
    attrs.reserve(fs.size());
    for (const auto& v : fs) {
      attrs.push_back(ir::FloatAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  ir::Attribute operator()(const std::vector<int>& is) {
    std::vector<ir::Attribute> attrs;
    attrs.reserve(is.size());
    for (const auto& v : is) {
      attrs.push_back(ir::Int32_tAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  ir::Attribute operator()(const std::vector<bool>& bs) {
    std::vector<ir::Attribute> attrs;
    attrs.reserve(bs.size());
    for (const auto& v : bs) {
      attrs.push_back(ir::BoolAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  ir::Attribute operator()(const std::vector<int64_t>& i64s) {
    std::vector<ir::Attribute> attrs;
    attrs.reserve(i64s.size());
    for (const auto& v : i64s) {
      attrs.push_back(ir::Int64_tAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  ir::Attribute operator()(const std::vector<double>& ds) {
    std::vector<ir::Attribute> attrs;
    attrs.reserve(ds.size());
    for (const auto& v : ds) {
      attrs.push_back(ir::DoubleAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  ir::Attribute operator()(
      const std::vector<paddle::experimental::Scalar>& ss) {
    std::vector<ir::Attribute> attrs;
    attrs.reserve(ss.size());
    for (const auto& v : ss) {
      attrs.push_back(paddle::dialect::ScalarAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
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
