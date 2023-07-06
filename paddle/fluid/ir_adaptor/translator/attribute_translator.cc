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

#include "paddle/fluid/ir_adaptor/translator/attribute_translator.h"

#include <string>
#include <vector>

#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/utils/variant.h"

namespace paddle {
namespace translator {

class AttributeVisitor {
 public:
  ir::IrContext* ctx;
  AttributeVisitor() { ctx = ir::IrContext::Instance(); }
  ~AttributeVisitor() {}

 public:
  virtual ir::Attribute operator()(int i) {
    VLOG(10) << "translating int";
    return ir::Int32Attribute::get(ctx, i);
  }

  virtual ir::Attribute operator()(int64_t i) {
    VLOG(10) << "translating int";
    return ir::Int64Attribute::get(ctx, i);
  }

  virtual ir::Attribute operator()(float f) {
    VLOG(10) << "translating float";
    return ir::FloatAttribute::get(ctx, f);
  }

  virtual ir::Attribute operator()(bool b) {
    VLOG(10) << "translating bool";
    return ir::BoolAttribute::get(ctx, b);
  }

  virtual ir::Attribute operator()(double d) {
    VLOG(10) << "translating double";
    return ir::DoubleAttribute::get(ctx, d);
  }

  virtual ir::Attribute operator()(const std::string& str) {
    VLOG(10) << "translating string";
    return ir::StrAttribute::get(ctx, str);
  }

  virtual ir::Attribute operator()(const paddle::experimental::Scalar& scalar) {
    VLOG(10) << "translating scalar";
    IR_THROW("not support translating paddle::experimental::Scalar");
  }

  virtual ir::Attribute operator()(const std::vector<std::string>& strs) {
    VLOG(10) << "translating vector<string>";
    std::vector<ir::Attribute> attrs;
    attrs.reserve(strs.size());
    for (const auto& v : strs) {
      attrs.push_back(ir::StrAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  virtual ir::Attribute operator()(const std::vector<float>& fs) {
    VLOG(10) << "translating vector<float>";
    std::vector<ir::Attribute> attrs;
    attrs.reserve(fs.size());
    for (const auto& v : fs) {
      attrs.push_back(ir::FloatAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  virtual ir::Attribute operator()(const std::vector<int>& is) {
    VLOG(10) << "translating vector<int>";
    std::vector<ir::Attribute> attrs;
    attrs.reserve(is.size());
    for (const auto& v : is) {
      attrs.push_back(ir::Int32Attribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  virtual ir::Attribute operator()(const std::vector<bool>& bs) {
    VLOG(10) << "translating vector<bool>";
    std::vector<ir::Attribute> attrs;
    attrs.reserve(bs.size());
    for (const auto& v : bs) {
      attrs.push_back(ir::BoolAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  virtual ir::Attribute operator()(const std::vector<int64_t>& i64s) {
    VLOG(10) << "translating vector<int64>";
    std::vector<ir::Attribute> attrs;
    attrs.reserve(i64s.size());
    for (const auto& v : i64s) {
      attrs.push_back(ir::Int64Attribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  virtual ir::Attribute operator()(const std::vector<double>& ds) {
    VLOG(10) << "translating vector<double>";
    std::vector<ir::Attribute> attrs;
    attrs.reserve(ds.size());
    for (const auto& v : ds) {
      attrs.push_back(ir::DoubleAttribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }

  virtual ir::Attribute operator()(
      const std::vector<paddle::experimental::Scalar>& ss) {
    VLOG(10) << "translating vector<scalar>";
    IR_THROW(
        "not support translating std::vector<paddle::experimental::Scalar>");
  }

  virtual ir::Attribute operator()(const paddle::blank& blank) {
    VLOG(10) << "translating paddle::blank";
    return ir::Attribute(nullptr);
  }

  template <typename T>
  ir::Attribute operator()(T attr) {
    VLOG(10) << "translating null type";
    return ir::Attribute(nullptr);
  }
};

class Int64ArrayAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;

  ir::Attribute operator()(const std::vector<int>& is) override {
    VLOG(10) << "translating vector<int64>";
    std::vector<ir::Attribute> attrs;
    attrs.reserve(is.size());
    for (const auto& v : is) {
      attrs.push_back(ir::Int64Attribute::get(ctx, v));
    }
    return ir::ArrayAttribute::get(ctx, attrs);
  }
};

class IntArrayAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;
  ir::Attribute operator()(const std::vector<int>& is) override {
    VLOG(10) << "translating vector<int> to IntArray";
    phi::IntArray data(is);
    return paddle::dialect::IntArrayAttribute::get(ctx, data);
  }

  ir::Attribute operator()(const std::vector<int64_t>& is) override {
    VLOG(10) << "translating vector<int> to IntArray";
    phi::IntArray data(is);
    return paddle::dialect::IntArrayAttribute::get(ctx, data);
  }
};

class DataTypeAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;
  ir::Attribute operator()(int i) override {
    VLOG(10) << "translating int to DataType: " << i;

    auto phi_dtype = phi::TransToPhiDataType(i);
    return paddle::dialect::DataTypeAttribute::get(ctx, phi_dtype);
  }

  ir::Attribute operator()(const paddle::blank& blank) override {
    VLOG(10) << "translating paddle::blank to DataType::UNDEFINED";
    return paddle::dialect::DataTypeAttribute::get(ctx, phi::DataType());
  }
};

class PlaceAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;

  ir::Attribute operator()(const paddle::blank& blank) override {
    VLOG(10) << "translating paddle::blank to Place::UNDEFINED";
    phi::Place data(phi::AllocationType::UNDEFINED);
    return paddle::dialect::PlaceAttribute::get(ctx, data);
  }
};

AttributeTranslator::AttributeTranslator() {
  general_visitor = new AttributeVisitor();
  special_visitors["paddle::dialect::IntArrayAttribute"] =
      new IntArrayAttributeVisitor();
  special_visitors["paddle::dialect::DataTypeAttribute"] =
      new DataTypeAttributeVisitor();
  special_visitors["paddle::dialect::PlaceAttribute"] =
      new PlaceAttributeVisitor();
  special_visitors["ir::ArrayAttribute<ir::Int64Attribute>"] =
      new Int64ArrayAttributeVisitor();
}

ir::Attribute AttributeTranslator::operator()(
    const framework::Attribute& attr) {
  return paddle::visit(*general_visitor, attr);
}

ir::Attribute AttributeTranslator::operator()(
    const std::string& target_type, const framework::Attribute& attr) {
  if (special_visitors.find(target_type) == special_visitors.end()) {
    VLOG(10) << "[" << target_type << "] not found";
    return paddle::visit(*general_visitor, attr);
  }
  return paddle::visit(*(special_visitors.at(target_type)), attr);
}

}  // namespace translator
}  // namespace paddle
