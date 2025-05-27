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
#include "paddle/ap/include/axpr/type.h"

#include "paddle/ap/include/drr/ir_op.h"
#include "paddle/ap/include/drr/ir_value.h"
#include "paddle/ap/include/drr/node.h"
#include "paddle/ap/include/drr/op_pattern_ctx.h"
#include "paddle/ap/include/drr/tags.h"
#include "paddle/ap/include/drr/tensor_pattern_ctx.h"
#include "paddle/ap/include/drr/unbound_ir_value.h"
#include "paddle/ap/include/drr/unbound_native_ir_op.h"
#include "paddle/ap/include/drr/unbound_opt_packed_ir_op.h"
#include "paddle/ap/include/drr/unbound_packed_ir_op.h"
#include "paddle/ap/include/drr/unbound_packed_ir_value.h"

namespace ap::drr {

struct OpTensorPatternCtxHelper {
  using OpPtnCtx = OpPatternCtx;
  using TensorPtnCtx = TensorPatternCtx;

  template <typename IrValueT>
  adt::Result<std::optional<axpr::Value>> GetOptType(const IrValueT& ir_value) {
    ADT_LET_CONST_REF(tensor_pattern_ctx,
                      adt::WeakPtrLock(ir_value->tensor_pattern_ctx));
    const auto& iter = tensor_pattern_ctx->uid2type.find(ir_value->name);
    if (iter == tensor_pattern_ctx->uid2type.end()) {
      return std::nullopt;
    }
    return iter->second;
  }

  template <typename IrValueT>
  adt::Result<adt::Ok> SetType(const IrValueT& ir_value,
                               const axpr::Value& type) {
    ADT_LET_CONST_REF(tensor_pattern_ctx,
                      adt::WeakPtrLock(ir_value->tensor_pattern_ctx));
    tensor_pattern_ctx->uid2type[ir_value->name] = type;
    return adt::Ok{};
  }

  adt::Result<axpr::Value> ConnectIrOpAndIrValue(
      const NativeIrOp<drr::Node>& native_ir_op,
      const adt::List<NativeIrValue<drr::Node>>& inputs,
      const adt::List<NativeIrValue<drr::Node>>& outputs) {
    ADT_LET_CONST_REF(op_upstream_nodes, native_ir_op->node.UpstreamNodes());
    ADT_CHECK(op_upstream_nodes.size() == 0);
    ADT_LET_CONST_REF(op_downstream_nodes,
                      native_ir_op->node.DownstreamNodes());
    ADT_CHECK(op_downstream_nodes.size() == 0);
    ADT_LET_CONST_REF(
        op_pattern_ctx,
        adt::WeakPtrLock(native_ir_op->op_declare->op_pattern_ctx));
    const auto& node_arena = op_pattern_ctx->node_arena;
    for (size_t i = 0; i < inputs->size(); ++i) {
      const auto& native_ir_op_operand = node_arena->New([&](const auto& node) {
        return NativeIrOpOperand<drr::Node>{node, i};
      });
      ADT_RETURN_IF_ERR(
          inputs->at(i)->node.ConnectTo(native_ir_op_operand.node(),
                                        graph::UnindexedTag<std::monostate>{},
                                        graph::IndexedTag<std::monostate>{}));
      ADT_RETURN_IF_ERR(native_ir_op_operand.node().ConnectTo(
          native_ir_op->node,
          graph::IndexedTag<std::monostate>{},
          graph::IndexedTag<std::monostate>{}));
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
      ADT_LET_CONST_REF(output_upstream_nodes,
                        outputs->at(i)->node.UpstreamNodes());
      ADT_CHECK(output_upstream_nodes.size() == 0);
      const auto& native_ir_op_result = node_arena->New([&](const auto& node) {
        return NativeIrOpResult<drr::Node>{node, i};
      });
      ADT_RETURN_IF_ERR(
          native_ir_op->node.ConnectTo(native_ir_op_result.node(),
                                       graph::IndexedTag<std::monostate>{},
                                       graph::IndexedTag<std::monostate>{}));
      ADT_RETURN_IF_ERR(native_ir_op_result.node().ConnectTo(
          outputs->at(i)->node,
          graph::IndexedTag<std::monostate>{},
          graph::IndexedTag<std::monostate>{}));
    }
    SetIrOpByUid(op_pattern_ctx, native_ir_op->name, native_ir_op);
    return adt::Nothing{};
  }

  adt::Result<axpr::Value> ConnectIrOpAndIrValue(
      const PackedIrOp<drr::Node>& packed_ir_op,
      const adt::List<IrValue>& inputs,
      const adt::List<IrValue>& outputs) {
    ADT_LET_CONST_REF(op_upstream_nodes, packed_ir_op->node.UpstreamNodes());
    ADT_CHECK(op_upstream_nodes.size() == 0);
    ADT_LET_CONST_REF(op_downstream_nodes,
                      packed_ir_op->node.DownstreamNodes());
    ADT_CHECK(op_downstream_nodes.size() == 0);
    ADT_LET_CONST_REF(
        op_pattern_ctx,
        adt::WeakPtrLock(packed_ir_op->op_declare->op_pattern_ctx));
    const auto& node_arena = op_pattern_ctx->node_arena;
    for (size_t i = 0; i < inputs->size(); ++i) {
      const auto& packed_ir_op_operand = node_arena->New([&](const auto& node) {
        return PackedIrOpOperand<drr::Node>{node, i};
      });
      ADT_RETURN_IF_ERR(
          inputs->at(i).node().ConnectTo(packed_ir_op_operand.node(),
                                         graph::UnindexedTag<std::monostate>{},
                                         graph::IndexedTag<std::monostate>{}));
      ADT_RETURN_IF_ERR(packed_ir_op_operand.node().ConnectTo(
          packed_ir_op->node,
          graph::IndexedTag<std::monostate>{},
          graph::UnindexedTag<std::monostate>{}));
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
      ADT_LET_CONST_REF(output_upstream_nodes,
                        outputs->at(i).node().UpstreamNodes());
      ADT_CHECK(output_upstream_nodes.size() == 0);
      const auto& packed_ir_op_result = node_arena->New([&](const auto& node) {
        return PackedIrOpResult<drr::Node>{node, i};
      });
      ADT_RETURN_IF_ERR(
          packed_ir_op->node.ConnectTo(packed_ir_op_result.node(),
                                       graph::UnindexedTag<std::monostate>{},
                                       graph::IndexedTag<std::monostate>{}));
      ADT_RETURN_IF_ERR(packed_ir_op_result.node().ConnectTo(
          outputs->at(i).node(),
          graph::IndexedTag<std::monostate>{},
          graph::IndexedTag<std::monostate>{}));
    }
    SetIrOpByUid(op_pattern_ctx, packed_ir_op->name, packed_ir_op);
    return adt::Nothing{};
  }

  adt::Result<axpr::Value> ConnectIrOpAndIrValue(
      const OptPackedIrOp<drr::Node>& packed_ir_op,
      const adt::List<IrValue>& inputs,
      const adt::List<IrValue>& outputs) {
    ADT_LET_CONST_REF(op_upstream_nodes, packed_ir_op->node.UpstreamNodes());
    ADT_CHECK(op_upstream_nodes.size() == 0);
    ADT_LET_CONST_REF(op_downstream_nodes,
                      packed_ir_op->node.DownstreamNodes());
    ADT_CHECK(op_downstream_nodes.size() == 0);
    ADT_LET_CONST_REF(
        op_pattern_ctx,
        adt::WeakPtrLock(packed_ir_op->op_declare->op_pattern_ctx));
    const auto& node_arena = op_pattern_ctx->node_arena;
    for (size_t i = 0; i < inputs->size(); ++i) {
      const auto& packed_ir_op_operand = node_arena->New([&](const auto& node) {
        return OptPackedIrOpOperand<drr::Node>{node, i};
      });
      ADT_RETURN_IF_ERR(
          inputs->at(i).node().ConnectTo(packed_ir_op_operand.node(),
                                         graph::UnindexedTag<std::monostate>{},
                                         graph::IndexedTag<std::monostate>{}));
      ADT_RETURN_IF_ERR(packed_ir_op_operand.node().ConnectTo(
          packed_ir_op->node,
          graph::IndexedTag<std::monostate>{},
          graph::UnindexedTag<std::monostate>{}));
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
      ADT_LET_CONST_REF(output_upstream_nodes,
                        outputs->at(i).node().UpstreamNodes());
      ADT_CHECK(output_upstream_nodes.size() == 0);
      const auto& packed_ir_op_result = node_arena->New([&](const auto& node) {
        return OptPackedIrOpResult<drr::Node>{node, i};
      });
      ADT_RETURN_IF_ERR(
          packed_ir_op->node.ConnectTo(packed_ir_op_result.node(),
                                       graph::UnindexedTag<std::monostate>{},
                                       graph::IndexedTag<std::monostate>{}));
      ADT_RETURN_IF_ERR(packed_ir_op_result.node().ConnectTo(
          outputs->at(i).node(),
          graph::IndexedTag<std::monostate>{},
          graph::IndexedTag<std::monostate>{}));
    }
    SetIrOpByUid(op_pattern_ctx, packed_ir_op->name, packed_ir_op);
    return adt::Nothing{};
  }

  template <typename OpPtnCtxT>
  adt::Result<IrOp> GetIrOpByUid(const OpPtnCtxT& self,
                                 const std::string& name) {
    const auto& iter = self->uid2ir_op.find(name);
    if (iter == self->uid2ir_op.end()) {
      return adt::errors::AttributeError{std::string() + "no op named '" +
                                         name + "' registered."};
    }
    return iter->second;
  }

  template <typename OpPtnCtxT>
  adt::Result<adt::Ok> CheckIrOpNameByUid(const OpPtnCtxT& self,
                                          const std::string& name,
                                          const IrOp& ir_op) {
    ADT_LET_CONST_REF(existed_ir_op, GetIrOpByUid(self, name));
    ADT_CHECK(ir_op.op_name() == existed_ir_op.op_name())
        << adt::errors::TypeError{
               std::string() + "CheckIrOpNameByUid() failed. lhs: " +
               ir_op.op_name() + ", rhs: " + existed_ir_op.op_name() + ""};
    return adt::Ok{};
  }

  template <typename OpPtnCtxT>
  bool HasIrOpByUid(const OpPtnCtxT& self, const std::string& name) {
    return self->uid2ir_op.count(name) > 0;
  }

  template <typename OpPtnCtxT>
  void SetIrOpByUid(const OpPtnCtxT& self,
                    const std::string& name,
                    const IrOp& ir_op) {
    self->uid2ir_op[name] = ir_op;
  }

  template <typename TensorPtnCtxT>
  bool HasIrValueByUid(const TensorPtnCtxT& self, const std::string& name) {
    return self->uid2ir_value.count(name);
  }

  template <typename TensorPtnCtxT>
  adt::Result<IrValue> GetIrValueByUid(const TensorPtnCtxT& self,
                                       const std::string& name) {
    const auto& iter = self->uid2ir_value.find(name);
    if (iter == self->uid2ir_value.end()) {
      return adt::errors::AttributeError{std::string() + "no tensor named '" +
                                         name + "' registered."};
    }
    return iter->second;
  }

  template <typename TensorPtnCtxT>
  void SetIrValueByUid(const TensorPtnCtxT& self,
                       const std::string& name,
                       const IrValue& ir_value) {
    self->uid2ir_value[name] = ir_value;
  }

  adt::Result<NativeIrValue<drr::Node>> CloneIrValueDataAndRegister(
      const TensorPtnCtx& self,
      const NativeIrValue<drr::Node>& native_ir_value) {
    const auto& cloned_node = self->node_arena->New([&](const auto& node) {
      return NativeIrValue<drr::Node>{
          node, native_ir_value->name, self.shared_ptr()};
    });
    ADT_CHECK(cloned_node.template Has<NativeIrValue<drr::Node>>());
    const auto& cloned = cloned_node.template Get<NativeIrValue<drr::Node>>();
    SetIrValueByUid(self, native_ir_value->name, cloned);
    return cloned;
  }

  adt::Result<PackedIrValue<drr::Node>> CloneIrValueDataAndRegister(
      const TensorPtnCtx& self,
      const PackedIrValue<drr::Node>& packed_ir_value) {
    const auto& cloned_node = self->node_arena->New([&](const auto& node) {
      return PackedIrValue<drr::Node>{node, packed_ir_value->name};
    });
    ADT_CHECK(cloned_node.template Has<PackedIrValue<drr::Node>>());
    const auto& cloned = cloned_node.template Get<PackedIrValue<drr::Node>>();
    SetIrValueByUid(self, packed_ir_value->name, cloned);
    return cloned;
  }

  adt::Result<NativeIrOp<drr::Node>> GetNativeIrOpByUnboundNativeIrOp(
      const UnboundNativeIrOp<drr::Node>& ir_op) {
    ADT_LET_CONST_REF(op_pattern_ctx,
                      adt::WeakPtrLock(ir_op->op_declare->op_pattern_ctx));
    const auto& node = op_pattern_ctx->node_arena->New([&](const auto& node) {
      return NativeIrOp<drr::Node>{node, ir_op->op_declare, ir_op->name};
    });
    ADT_CHECK(node.template Has<NativeIrOp<drr::Node>>());
    return node.template Get<NativeIrOp<drr::Node>>();
  }

  adt::Result<PackedIrOp<drr::Node>> GetPackedIrOpByUnboundPackedIrOp(
      const UnboundPackedIrOp<drr::Node>& ir_op) {
    ADT_LET_CONST_REF(op_pattern_ctx,
                      adt::WeakPtrLock(ir_op->op_declare->op_pattern_ctx));
    const auto& node = op_pattern_ctx->node_arena->New([&](const auto& node) {
      return PackedIrOp<drr::Node>{node, ir_op->op_declare, ir_op->name};
    });
    ADT_CHECK(node.template Has<PackedIrOp<drr::Node>>());
    return node.template Get<PackedIrOp<drr::Node>>();
  }

  adt::Result<OptPackedIrOp<drr::Node>> GetOptPackedIrOpByUnboundOptPackedIrOp(
      const UnboundOptPackedIrOp<drr::Node>& ir_op) {
    ADT_LET_CONST_REF(op_pattern_ctx,
                      adt::WeakPtrLock(ir_op->op_declare->op_pattern_ctx));
    const auto& node = op_pattern_ctx->node_arena->New([&](const auto& node) {
      return OptPackedIrOp<drr::Node>{node, ir_op->op_declare, ir_op->name};
    });
    return node.template TryGet<OptPackedIrOp<drr::Node>>();
  }

  adt::Result<NativeIrValue<drr::Node>> GetNativeIrValueByUnboundIrValue(
      const UnboundIrValue<drr::Node>& unbound_ir_value) {
    ADT_LET_CONST_REF(tensor_ctx,
                      adt::WeakPtrLock(unbound_ir_value->tensor_pattern_ctx));
    if (HasIrValueByUid(tensor_ctx, unbound_ir_value->name)) {
      ADT_LET_CONST_REF(ir_value,
                        GetIrValueByUid(tensor_ctx, unbound_ir_value->name));
      const auto& opt_ret = ir_value.Match(
          [](const NativeIrValue<drr::Node>& impl)
              -> adt::Result<NativeIrValue<drr::Node>> { return impl; },
          [&](const auto&) -> adt::Result<NativeIrValue<drr::Node>> {
            return adt::errors::RuntimeError{"only NativeIrValue supported."};
          });
      ADT_LET_CONST_REF(ret, opt_ret);
      return ret;
    }
    const auto& node_arena = tensor_ctx->node_arena;
    const auto& node = node_arena->New([&](const auto& node) {
      return NativeIrValue<drr::Node>{
          node, unbound_ir_value->name, unbound_ir_value->tensor_pattern_ctx};
    });
    ADT_CHECK(node.template Has<NativeIrValue<drr::Node>>());
    const auto& native_ir_value = node.template Get<NativeIrValue<drr::Node>>();
    SetIrValueByUid(tensor_ctx, native_ir_value->name, native_ir_value);
    return native_ir_value;
  }

  adt::Result<PackedIrValue<drr::Node>> GetPackedIrValueByUnboundPackedIrValue(
      const UnboundPackedIrValue<drr::Node>& ir_value) {
    ADT_LET_CONST_REF(tensor_ctx,
                      adt::WeakPtrLock(ir_value->tensor_pattern_ctx));
    if (HasIrValueByUid(tensor_ctx, ir_value->name)) {
      ADT_LET_CONST_REF(ir_value, GetIrValueByUid(tensor_ctx, ir_value->name));
      const auto& opt_ret = ir_value.Match(
          [](const PackedIrValue<drr::Node>& impl)
              -> adt::Result<PackedIrValue<drr::Node>> { return impl; },
          [&](const auto&) -> adt::Result<PackedIrValue<drr::Node>> {
            return adt::errors::RuntimeError{"only PackedIrValue supported."};
          });
      ADT_LET_CONST_REF(ret, opt_ret);
      return ret;
    }
    const auto& node_arena = tensor_ctx->node_arena;
    const auto& node = node_arena->New([&](const auto& node) {
      return PackedIrValue<drr::Node>{node, ir_value->name};
    });
    ADT_CHECK(node.template Has<PackedIrValue<drr::Node>>());
    const auto& packed_ir_value = node.template Get<PackedIrValue<drr::Node>>();
    SetIrValueByUid(tensor_ctx, packed_ir_value->name, packed_ir_value);
    return packed_ir_value;
  }
};

}  // namespace ap::drr
