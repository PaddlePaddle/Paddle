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
#include "paddle/ap/include/ir_match/ir_match_ctx.h"
#include "paddle/ap/include/ir_match/tensor_match_ctx.h"

namespace ap::ir_match {

template <typename ValueT, typename BirNode>
struct TensorMatchCtxMethodClass {
  using This = TensorMatchCtxMethodClass;
  using Self = ir_match::TensorMatchCtx<BirNode>;

  static adt::Result<ValueT> GetAttr(const ValueT& self_val,
                                     const std::vector<ValueT>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    const auto& attr_name_val = args.at(0);
    ADT_LET_CONST_REF(attr_name, axpr::TryGetImpl<std::string>(attr_name_val));
    ADT_LET_CONST_REF(ir_tensor, This{}.GetIrTensorByName(self, attr_name));
    if (ir_tensor.has_value()) {
      return ir_tensor.value();
    }
    return adt::errors::TypeError{std::string() +
                                  "'TensorMatchCtx' has no attribute '" +
                                  attr_name + "'"};
  }

  using DrrValueT = drr::Value;
  using DrrNodeT = drr::Node;
  using DrrNativeIrValue = drr::NativeIrValue<DrrNodeT>;
  using DrrPackedIrValue = drr::PackedIrValue<DrrNodeT>;
  using SmallGraphNodeT = graph::Node<DrrNodeT>;

  using IrNativeIrValue = typename BirNode::native_value_type;
  using IrPackedIrValue = typename BirNode::packed_value_type;
  using IrRefIrValue = typename BirNode::ref_value_type;

  adt::Result<std::optional<ValueT>> GetIrTensorByName(
      const Self& self, const std::string& attr_name) {
    ADT_LET_CONST_REF(ir_match_ctx, adt::WeakPtrLock(self->ir_mtach_ctx));
    const auto& source_pattern_ctx = ir_match_ctx->source_pattern_ctx;
    const auto& tensor_pattern_ctx = source_pattern_ctx->tensor_pattern_ctx;
    const auto& iter = tensor_pattern_ctx->uid2ir_value.find(attr_name);
    if (iter == tensor_pattern_ctx->uid2ir_value.end()) {
      return std::nullopt;
    }
    using RetT = adt::Result<ValueT>;
    auto GetNativeIrValueBySmallGraphNode =
        [&](const SmallGraphNodeT& node) -> RetT {
      const auto& graph_match_ctx = ir_match_ctx->graph_match_ctx;
      ADT_LET_CONST_REF(bir_value_node,
                        graph_match_ctx->GetSoleBigGraphNode(node));
      return CastFromBirValue(bir_value_node);
    };
    auto GetPackedIrValuesBySmallGraphNode =
        [&](const SmallGraphNodeT& node) -> RetT {
      const auto& graph_match_ctx = ir_match_ctx->graph_match_ctx;
      ADT_LET_CONST_REF(bir_nodes,
                        graph_match_ctx->GetPackedBigGraphIrValueNodes(node));
      adt::List<ValueT> ret;
      ret->reserve(bir_nodes->size());
      for (const auto& bir_node : *bir_nodes) {
        ADT_LET_CONST_REF(elt, CastFromBirValue(bir_node));
        ret->emplace_back(elt);
      }
      return ret;
    };
    ADT_LET_CONST_REF(
        ir_value,
        iter->second.Match(
            [&](const DrrNativeIrValue& native_ir_value) -> RetT {
              return GetNativeIrValueBySmallGraphNode(native_ir_value->node);
            },
            [&](const DrrPackedIrValue& packed_ir_value) -> RetT {
              return GetPackedIrValuesBySmallGraphNode(packed_ir_value->node);
            },
            [&](const auto&) -> RetT {
              return adt::errors::ValueError{
                  std::string() + "Failed to get OpMatchCtx attribute, '" +
                  attr_name + "' is a unbounded op which should not be."};
            }));
    return ir_value;
  }

  adt::Result<ValueT> CastFromBirValue(const BirNode& bir_value_node) {
    return bir_value_node.Match(
        [&](const IrNativeIrValue& impl) -> adt::Result<ValueT> {
          axpr::BuiltinClassInstance<ValueT> instance{
              impl.template GetBuiltinClass<ValueT>(), impl};
          return ValueT{instance};
        },
        [&](const IrRefIrValue& impl) -> adt::Result<ValueT> {
          axpr::BuiltinClassInstance<ValueT> instance{
              impl.template GetBuiltinClass<ValueT>(), impl};
          return ValueT{instance};
        },
        [&](const auto&) -> adt::Result<ValueT> {
          return adt::errors::RuntimeError{
              std::string() +
              "a drr op node has wrongly matched to a non-op ir node."};
        });
  }
};

template <typename ValueT, typename BirNode>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> GetTensorMatchCtxClass() {
  using ImplMethods = TensorMatchCtxMethodClass<ValueT, BirNode>;
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("TensorMatchCtx", [&](const auto& Define) {
        Define("__getattr__", &ImplMethods::GetAttr);
      }));
  return axpr::MakeGlobalNaiveClassOps<typename ImplMethods::Self>(cls);
}

}  // namespace ap::ir_match
