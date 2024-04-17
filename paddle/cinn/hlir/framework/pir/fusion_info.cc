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

#include "paddle/cinn/hlir/framework/pir/fusion_info.h"
#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/pir/include/core/ir_printer.h"
PD_DECLARE_bool(enable_cinn_compile_cache);

namespace cinn::hlir::framework::pir {

constexpr static char* kOpCallStack = "op_callstack";

std::size_t AttributeInfo::hash() const { return attr_.hash(); }

std::ostream& operator<<(std::ostream& os, const AttributeInfo& attr_info) {
  os << "AttributeInfo - " << attr_info.name_ << ", " << attr_info.hash();
  if (VLOG_IS_ON(7)) {
    os << " (";
    ::pir::IrPrinter(os).PrintAttribute(attr_info.attr_);
    os << ")";
  }
  return os;
}

std::size_t ValueInfo::hash() const { return type_.hash(); }

std::ostream& operator<<(std::ostream& os, const ValueInfo& value_info) {
  os << "ValueInfo - " << value_info.hash();
  if (VLOG_IS_ON(7)) {
    os << "(";
    ::pir::IrPrinter(os).PrintType(value_info.type_);
    os << ")";
  }
  return os;
}

OperationInfo::OperationInfo(const ::pir::Operation& op) {
  name_ = op.name();
  input_infos_.reserve(op.num_operands());
  for (const auto value : op.operands_source()) {
    if (!value || !value.type()) continue;
    input_infos_.emplace_back(value);
  }
  output_infos_.reserve(op.num_results());
  for (const auto value : op.results()) {
    if (!value || !value.type()) continue;
    output_infos_.emplace_back(value);
  }
  // Keep attribute always in order.
  const auto& attributes = op.attributes();
  std::map<std::string, ::pir::Attribute, std::less<>> order_attributes(
      attributes.begin(), attributes.end());
  attr_infos_.reserve(attributes.size());
  for (const auto& [attr_name, attr_value] : order_attributes) {
    if (!attr_value || attr_name == kOpCallStack) continue;
    attr_infos_.emplace_back(attr_name, attr_value);
  }
}

std::size_t OperationInfo::hash() const {
  std::size_t seed = 1789;
  hash_combine(seed, name_);
  for (const auto& info : input_infos_) hash_combine(seed, info);
  for (const auto& info : output_infos_) hash_combine(seed, info);
  for (const auto& info : attr_infos_) hash_combine(seed, info);
  return seed;
}

std::ostream& operator<<(std::ostream& os, const OperationInfo& op_info) {
  os << op_info.name_ << " - " << op_info.hash();
  if (VLOG_IS_ON(7)) {
    os << "{\n";
    for (const auto& info : op_info.input_infos_) os << info << "\n";
    for (const auto& info : op_info.output_infos_) os << info << "\n";
    for (const auto& info : op_info.attr_infos_) os << info << "\n";
    os << "}";
  }
  return os;
}

std::size_t FusionOpInfo::hash() const {
  std::size_t seed = op_info_.hash();
  for (const auto& [value_index, op_info_hash] : inner_deps_) {
    hash_combine(seed, value_index);
    hash_combine(seed, op_info_hash);
  }
  return seed;
}

std::ostream& operator<<(std::ostream& os, const FusionOpInfo& info) {
  os << info.op_info_ << ", inner_deps:{";
  for (const auto& [value_index, op_info_hash] : info.inner_deps_) {
    os << " (" << value_index << ", " << op_info_hash << ")";
  }
  os << "}";
  return os;
}

FusionInfo::FusionInfo(const OpLoweringGroup& group) {
  std::unordered_map<const ::pir::Operation*, size_t> op_mapper;
  unique_fn_name_ = group.FuncName();

  const auto GetInnerUpstreamOps =
      [&](const ::pir::Operation* op) -> decltype(auto) {
    std::unordered_map<size_t, size_t> upstream_ops_index_hash;
    for (size_t i = 0; i < op->num_operands(); ++i) {
      const auto value = op->operand_source(i);
      if (!value || !value.defining_op()) continue;
      const auto* defining_op = value.defining_op();
      if (op_mapper.count(defining_op) == 0) continue;
      PADDLE_ENFORCE_LT(op_mapper[defining_op],
                        this->op_infos_.size(),
                        ::common::errors::OutOfRange(
                            "Required op_mapper[defining_op] < "
                            "op_infos_.size(), but received index %d",
                            op_mapper[defining_op]));
      upstream_ops_index_hash.emplace(
          i, this->op_infos_[op_mapper[defining_op]].hash());
    }
    return upstream_ops_index_hash;
  };

  const auto sorted_ops = TopologySort(group);
  for (size_t i = 0; i < sorted_ops.size(); ++i) {
    const auto& op = sorted_ops[i];
    op_infos_.emplace_back(*op, GetInnerUpstreamOps(op));
    op_mapper.insert({op, i});
  }
}

std::size_t FusionInfo::hash() const {
  if (cached_hash_value_ != 0U) {
    return cached_hash_value_;
  }
  std::size_t seed = 2153;
  for (const auto& info : op_infos_) hash_combine(seed, info);
  if (!FLAGS_enable_cinn_compile_cache) hash_combine(seed, unique_fn_name_);
  return seed;
}

std::ostream& operator<<(std::ostream& os, const FusionInfo& fusion_info) {
  os << "FusionInfo - " << fusion_info.hash();
  if (VLOG_IS_ON(5)) {
    os << "{\n";
    if (!FLAGS_enable_cinn_compile_cache)
      os << "fn_name: " << fusion_info.unique_fn_name_;
    for (const auto& op_info : fusion_info.op_infos_) os << op_info << "\n";
    os << "}\n";
  }
  return os;
}

std::size_t HashIntArgsMap(
    const std::map<int, CINNKernelInfo::ArgDimIdx>& int_args_map) {
  std::size_t seed = 2153;
  for (const auto& [input_idx, dim_idx] : int_args_map) {
    hash_combine(seed, input_idx);
    hash_combine(seed, dim_idx.arg_idx);
    hash_combine(seed, dim_idx.dim_idx);
  }
  return seed;
}
std::ostream& operator<<(
    std::ostream& os,
    const std::map<int, CINNKernelInfo::ArgDimIdx>& int_args_map) {
  os << "int_args_map: {\n";
  for (const auto& [input_idx, dim_idx] : int_args_map) {
    os << "input_idx: " << input_idx << ":[ " << dim_idx.arg_idx << ", "
       << dim_idx.dim_idx << " ]\n";
  }
  os << "}\n";
}

std::vector<const ::pir::Operation*> TopologySort(
    const OpLoweringGroup& group) {
  // NOTE(Aurelius84): Use simplest one-by-one order temporaly.
  auto* block = group.GetParentBlock();
  std::vector<const ::pir::Operation*> ops;
  ops.reserve(block->size());
  for (auto& op : *block) {
    ops.push_back(&op);
  }
  return ops;
}

}  // namespace cinn::hlir::framework::pir
