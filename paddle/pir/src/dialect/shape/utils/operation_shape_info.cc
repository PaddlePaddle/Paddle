// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/include/dialect/shape/utils/operation_shape_info.h"
#include "paddle/pir/include/core/ir_printer.h"

namespace pir {
static const char *kOpCallStack = "op_callstack";
static const char *kSymShapeStr = "sym_shape_str";
static const char *kResultName = "name";

OperationShapeInfo::OperationShapeInfo(
    const Operation &op,
    const std::vector<symbol::ShapeOrDataDimExprs> &input_shape_or_datas)
    : OperationShapeInfo(op.name(), input_shape_or_datas, op.attributes()) {}

OperationShapeInfo::OperationShapeInfo(
    const std::string &op_name,
    const std::vector<symbol::ShapeOrDataDimExprs> &input_shape_or_datas,
    const AttributeMap &attributes)
    : op_name_(op_name), input_shape_or_datas_(input_shape_or_datas) {
  // Keep attribute always in order.
  std::map<std::string, ::pir::Attribute, std::less<>> order_attributes(
      attributes.begin(), attributes.end());
  attributes_.reserve(attributes.size());
  for (const auto &[attr_name, attr_value] : order_attributes) {
    if (!attr_value || attr_name == kOpCallStack || attr_name == kSymShapeStr ||
        attr_name == kResultName)
      continue;
    attributes_.emplace_back(attr_name, attr_value);
  }
}

std::size_t OperationShapeInfo::hash() const {
  const auto name_hash_func = std::hash<std::string>();
  const auto attr_hash_func = std::hash<pir::Attribute>();
  const auto shape_hash_func = std::hash<symbol::ShapeOrDataDimExprs>();
  std::size_t res = name_hash_func(op_name_);
  for (const auto &item : attributes_) {
    res = pir::detail::hash_combine(res, name_hash_func(item.first));
    res = pir::detail::hash_combine(res, attr_hash_func(item.second));
  }
  for (const auto &item : input_shape_or_datas_) {
    res = pir::detail::hash_combine(res, shape_hash_func(item));
  }
  return res;
}

bool OperationShapeInfo::operator==(const OperationShapeInfo &other) const {
  if (op_name_ != other.op_name_) return false;
  if (attributes_.size() != other.attributes_.size()) return false;
  for (std::size_t i = 0; i < attributes_.size(); ++i) {
    if (attributes_[i].first != other.attributes_[i].first ||
        attributes_[i].second != other.attributes_[i].second)
      return false;
  }
  if (input_shape_or_datas_.size() != other.input_shape_or_datas_.size())
    return false;
  for (std::size_t i = 0; i < input_shape_or_datas_.size(); ++i) {
    if (input_shape_or_datas_[i] != other.input_shape_or_datas_[i])
      return false;
  }
  return true;
}

std::ostream &operator<<(std::ostream &os, const OperationShapeInfo &info) {
  os << "OperationShapeInfo - " << info.op_name_ << std::endl;
  if (!info.attributes_.empty()) {
    os << "  attrs: {";
    for (std::size_t i = 0; i < info.attributes_.size() - 1; ++i) {
      ::pir::IrPrinter(os).PrintAttribute(info.attributes_[i].second);
      os << ", ";
    }
    ::pir::IrPrinter(os).PrintAttribute(info.attributes_.back().second);
    os << std::endl;
  }
  if (!info.input_shape_or_datas_.empty()) {
    os << "  input_shape_or_datas: {";
    for (std::size_t i = 0; i < info.input_shape_or_datas_.size() - 1; ++i) {
      os << info.input_shape_or_datas_[i] << ", ";
    }
    os << info.input_shape_or_datas_.back() << "}" << std::endl;
  }
  return os;
}

}  // namespace pir
