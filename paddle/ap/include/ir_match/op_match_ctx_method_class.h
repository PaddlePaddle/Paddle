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

#pragma once

#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/ir_match/ir_match_ctx.h"
#include "paddle/ap/include/ir_match/op_match_ctx.h"

namespace ap::ir_match {

template <typename ValueT, typename BirNode>
struct OpMatchCtxMethodClass {
  using This = OpMatchCtxMethodClass;
  using Self = ir_match::OpMatchCtx<BirNode>;

  static adt::Result<ValueT> GetAttr(const ValueT& self_val,
                                     const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    ADT_LET_CONST_REF(ir_op, This{}.GetIrOpByName(self, attr_name));
    if (ir_op.has_value()) {
      return ir_op.value();
    }
    return adt::errors::TypeError{
        std::string() + "'OpMatchCtx' has no attribute '" + attr_name + "'"};
  }

  using DrrNativeIrOp = drr::NativeIrOp<drr::Node>;
  using DrrPackedIrOp = drr::PackedIrOp<drr::Node>;
  using DrrOptPackedIrOp = drr::OptPackedIrOp<drr::Node>;
  using SmallGraphNodeT = graph::Node<drr::Node>;

  using IrNativeIrOp = typename BirNode::native_op_type;
  using IrPackedIrOp = typename BirNode::packed_op_type;
  using IrRefIrOp = typename BirNode::ref_op_type;

  adt::Result<std::optional<ValueT>> GetIrOpByName(
      const Self& self, const std::string& attr_name) {
    ADT_LET_CONST_REF(ir_match_ctx, adt::WeakPtrLock(self->ir_mtach_ctx));
    const auto& source_pattern_ctx = ir_match_ctx->source_pattern_ctx;
    const auto& op_pattern_ctx = source_pattern_ctx->op_pattern_ctx;
    const auto& iter = op_pattern_ctx->uid2ir_op.find(attr_name);
    if (iter == op_pattern_ctx->uid2ir_op.end()) {
      return std::nullopt;
    }
    auto GetIrOpBySmallGraphNode =
        [&](const SmallGraphNodeT& node) -> adt::Result<BirNode> {
      const auto& graph_match_ctx = ir_match_ctx->graph_match_ctx;
      return graph_match_ctx->GetSoleBigGraphNode(node);
    };
    ADT_LET_CONST_REF(
        ir_node,
        iter->second.Match(
            [&](const DrrNativeIrOp& native_ir_op) -> adt::Result<BirNode> {
              return GetIrOpBySmallGraphNode(native_ir_op->node);
            },
            [&](const DrrPackedIrOp& packed_ir_op) -> adt::Result<BirNode> {
              return GetIrOpBySmallGraphNode(packed_ir_op->node);
            },
            [&](const DrrOptPackedIrOp& packed_ir_op) -> adt::Result<BirNode> {
              return GetIrOpBySmallGraphNode(packed_ir_op->node);
            },
            [&](const auto&) -> adt::Result<BirNode> {
              return adt::errors::ValueError{
                  std::string() + "Failed to get OpMatchCtx attribute, '" +
                  attr_name + "' is a unbounded op which should not be."};
            }));
    ADT_LET_CONST_REF(
        ir_op,
        ir_node.Match(
            [&](const IrNativeIrOp& impl) -> adt::Result<ValueT> {
              axpr::BuiltinClassInstance<ValueT> instance{
                  impl.template GetBuiltinClass<ValueT>(), impl};
              return ValueT{instance};
            },
            [&](const IrPackedIrOp& impl) -> adt::Result<ValueT> {
              axpr::BuiltinClassInstance<ValueT> instance{
                  impl.template GetBuiltinClass<ValueT>(), impl};
              return ValueT{instance};
            },
            [&](const IrRefIrOp& impl) -> adt::Result<ValueT> {
              axpr::BuiltinClassInstance<ValueT> instance{
                  impl.template GetBuiltinClass<ValueT>(), impl};
              return ValueT{instance};
            },
            [&](const auto&) -> adt::Result<ValueT> {
              return adt::errors::RuntimeError{
                  std::string() +
                  "a ptn op node has wrongly matched to a non-op ir node."};
            }));
    return ir_op;
  }
};

template <typename ValueT, typename BirNode>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetOpMatchCtxClass() {
  using Impl = OpMatchCtxMethodClass<ValueT, BirNode>;
  static auto cls(axpr::MakeBuiltinClass<ValueT>(
      "OpMatchCtx",
      [&](const auto& Define) { Define("__getattr__", &Impl::GetAttr); }));
  return axpr::MakeGlobalNaiveClassOps<typename Impl::Self>(cls);
}

}  // namespace ap::ir_match
