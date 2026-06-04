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

#include "paddle/common/enforce.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/utils/variant.h"

namespace paddle::translator {

class AttributeVisitor {
 public:
  pir::IrContext* ctx;
  AttributeVisitor() { ctx = pir::IrContext::Instance(); }
  virtual ~AttributeVisitor() = default;

 public:
  virtual pir::Attribute operator()(int i) {
    VLOG(10) << "translating int";
    return pir::Int32Attribute::get(ctx, i);
  }

  virtual pir::Attribute operator()(int64_t i) {
    VLOG(10) << "translating int";
    return pir::Int64Attribute::get(ctx, i);
  }

  virtual pir::Attribute operator()(float f) {
    VLOG(10) << "translating float";
    return pir::FloatAttribute::get(ctx, f);
  }

  virtual pir::Attribute operator()(bool b) {
    VLOG(10) << "translating bool";
    return pir::BoolAttribute::get(ctx, b);
  }

  virtual pir::Attribute operator()(double d) {
    VLOG(10) << "translating double";
    return pir::DoubleAttribute::get(ctx, d);
  }

  virtual pir::Attribute operator()(const std::string& str) {
    VLOG(10) << "translating string";
    return pir::StrAttribute::get(ctx, str);
  }

  virtual pir::Attribute operator()(
      const paddle::experimental::Scalar& scalar) {
    VLOG(10) << "translating scalar";
    PADDLE_THROW(common::errors::Unimplemented(
        "not support "
        "translating paddle::experimental::Scalar"));
  }

  virtual pir::Attribute operator()(const std::vector<std::string>& strs) {
    VLOG(10) << "translating vector<string>";
    std::vector<pir::Attribute> attrs;
    attrs.reserve(strs.size());
    for (const auto& v : strs) {
      attrs.push_back(pir::StrAttribute::get(ctx, v));
    }
    return pir::ArrayAttribute::get(ctx, attrs);
  }

  virtual pir::Attribute operator()(const std::vector<float>& fs) {
    VLOG(10) << "translating vector<float>";
    std::vector<pir::Attribute> attrs;
    attrs.reserve(fs.size());
    for (const auto& v : fs) {
      attrs.push_back(pir::FloatAttribute::get(ctx, v));
    }
    return pir::ArrayAttribute::get(ctx, attrs);
  }

  virtual pir::Attribute operator()(const std::vector<int>& is) {
    VLOG(10) << "translating vector<int>";
    std::vector<pir::Attribute> attrs;
    attrs.reserve(is.size());
    for (const auto& v : is) {
      attrs.push_back(pir::Int32Attribute::get(ctx, v));
    }
    return pir::ArrayAttribute::get(ctx, attrs);
  }

  virtual pir::Attribute operator()(const std::vector<bool>& bs) {
    VLOG(10) << "translating vector<bool>";
    std::vector<pir::Attribute> attrs;
    attrs.reserve(bs.size());
    for (const auto& v : bs) {
      attrs.push_back(pir::BoolAttribute::get(ctx, v));
    }
    return pir::ArrayAttribute::get(ctx, attrs);
  }

  virtual pir::Attribute operator()(const std::vector<int64_t>& i64s) {
    VLOG(10) << "translating vector<int64> size: " << i64s.size();
    std::vector<pir::Attribute> attrs;
    attrs.reserve(i64s.size());
    for (const auto& v : i64s) {
      attrs.push_back(pir::Int64Attribute::get(ctx, v));
    }
    return pir::ArrayAttribute::get(ctx, attrs);
  }

  virtual pir::Attribute operator()(const std::vector<double>& ds) {
    VLOG(10) << "translating vector<double>";
    std::vector<pir::Attribute> attrs;
    attrs.reserve(ds.size());
    for (const auto& v : ds) {
      attrs.push_back(pir::DoubleAttribute::get(ctx, v));
    }
    return pir::ArrayAttribute::get(ctx, attrs);
  }

  virtual pir::Attribute operator()(
      const std::vector<paddle::experimental::Scalar>& ss) {
    VLOG(10) << "translating vector<scalar>";
    std::vector<pir::Attribute> attrs;
    attrs.reserve(ss.size());
    for (const auto& v : ss) {
      attrs.push_back(dialect::ScalarAttribute::get(ctx, v));
    }
    VLOG(10) << "translating vector<scalar> Done";
    return pir::ArrayAttribute::get(ctx, attrs);
  }

  virtual pir::Attribute operator()(const paddle::blank& blank) {
    VLOG(10) << "translating paddle::blank";
    return pir::Attribute(nullptr);
  }

  template <typename T>
  pir::Attribute operator()(T attr) {
    VLOG(10) << "translating null type";
    return pir::Attribute(nullptr);
  }
};

class Int64ArrayAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;

  pir::Attribute operator()(const std::vector<int>& is) override {
    VLOG(10) << "translating vector<int64>";
    std::vector<pir::Attribute> attrs;
    attrs.reserve(is.size());
    for (const auto& v : is) {
      attrs.push_back(pir::Int64Attribute::get(ctx, v));
    }
    return pir::ArrayAttribute::get(ctx, attrs);
  }

  pir::Attribute operator()(const paddle::blank& blank) override {
    VLOG(10) << "translating paddle::blank to int64[]";
    return pir::ArrayAttribute::get(ctx, {});
  }
};

class Int64AttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;

  pir::Attribute operator()(int is) override {
    VLOG(10) << "translating int to Int64Attribute";
    return pir::Int64Attribute::get(ctx, is);
  }
};

class Int32ArrayAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;

  pir::Attribute operator()(const std::vector<int64_t>& i64s) override {
    VLOG(10) << "translating vector<int64> size: " << i64s.size();
    std::vector<pir::Attribute> attrs;
    attrs.reserve(i64s.size());
    for (const auto& v : i64s) {
      attrs.push_back(pir::Int32Attribute::get(ctx, v));
    }
    return pir::ArrayAttribute::get(ctx, attrs);
  }
};

class IntArrayAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;
  pir::Attribute operator()(const std::vector<int>& is) override {
    VLOG(10) << "translating vector<int> to IntArray";
    phi::IntArray data(is);
    return paddle::dialect::IntArrayAttribute::get(ctx, data);
  }

  pir::Attribute operator()(const std::vector<int64_t>& is) override {
    VLOG(10) << "translating vector<int> to IntArray";
    phi::IntArray data(is);
    return paddle::dialect::IntArrayAttribute::get(ctx, data);
  }
};

class DataTypeAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;
  pir::Attribute operator()(int i) override {
    VLOG(10) << "translating int to DataType: " << i;

    auto phi_dtype = phi::TransToPhiDataType(i);
    return paddle::dialect::DataTypeAttribute::get(ctx, phi_dtype);
  }

  pir::Attribute operator()(const paddle::blank& blank) override {
    VLOG(10) << "translating paddle::blank to DataType::UNDEFINED";
    return paddle::dialect::DataTypeAttribute::get(ctx, phi::DataType());
  }
};

class PlaceAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;

  pir::Attribute operator()(const paddle::blank& blank) override {
    VLOG(10) << "translating paddle::blank to Place::UNDEFINED";
    phi::Place data(phi::AllocationType::UNDEFINED);
    return paddle::dialect::PlaceAttribute::get(ctx, data);
  }
};

class BoolAttributeVisitor : public AttributeVisitor {
 public:
  using AttributeVisitor::AttributeVisitor;

  pir::Attribute operator()(const paddle::blank& blank) override {
    VLOG(10) << "translating paddle::blank to Place::UNDEFINED";
    return pir::BoolAttribute::get(ctx, false);
  }

  pir::Attribute operator()(int i) override {
    VLOG(10) << "translating int";
    return pir::BoolAttribute::get(ctx, static_cast<bool>(i));
  }

  pir::Attribute operator()(int64_t i) override {
    VLOG(10) << "translating int64";
    return pir::BoolAttribute::get(ctx, static_cast<bool>(i));
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
  special_visitors["pir::ArrayAttribute<pir::Int64Attribute>"] =
      new Int64ArrayAttributeVisitor();
  special_visitors["pir::ArrayAttribute<pir::Int32Attribute>"] =
      new Int32ArrayAttributeVisitor();
  special_visitors["pir::Int64Attribute"] = new Int64AttributeVisitor();
  special_visitors["pir::BoolAttribute"] = new BoolAttributeVisitor();
}

pir::Attribute AttributeTranslator::operator()(
    const framework::Attribute& attr) {
  return paddle::visit(*general_visitor, attr);
}

pir::Attribute AttributeTranslator::operator()(
    const std::string& target_type, const framework::Attribute& attr) {
  if (special_visitors.find(target_type) == special_visitors.end()) {
    VLOG(10) << "[" << target_type << "] not found";
    return paddle::visit(*general_visitor, attr);
  }
  return paddle::visit(*(special_visitors.at(target_type)), attr);
}

}  // namespace paddle::translator
