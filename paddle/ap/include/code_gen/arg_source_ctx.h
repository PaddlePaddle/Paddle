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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace ap::code_gen {

struct NativeIrValueSource {
  int native_ir_value_index;

  bool operator==(const NativeIrValueSource& other) const {
    return this->native_ir_value_index == other.native_ir_value_index;
  }
};

struct PackedIrValueSource {
  int packed_ir_value_index;
  int tensor_member_index;

  bool operator==(const PackedIrValueSource& other) const {
    return this->packed_ir_value_index == other.packed_ir_value_index &&
           this->tensor_member_index == other.tensor_member_index;
  }
};

using TensorSourceImpl = std::variant<NativeIrValueSource, PackedIrValueSource>;

struct TensorSource : public TensorSourceImpl {
  using TensorSourceImpl::TensorSourceImpl;
  ADT_DEFINE_VARIANT_METHODS(TensorSourceImpl);
};

struct InTensorSource {
  TensorSource tensor_source;

  bool operator==(const InTensorSource& other) const {
    return this->tensor_source == other.tensor_source;
  }
};

struct OutTensorSource {
  TensorSource tensor_source;

  bool operator==(const OutTensorSource& other) const {
    return this->tensor_source == other.tensor_source;
  }
};

using InOutTensorSourceImpl = std::variant<InTensorSource, OutTensorSource>;

struct InOutTensorSource : public InOutTensorSourceImpl {
  using InOutTensorSourceImpl::InOutTensorSourceImpl;
  ADT_DEFINE_VARIANT_METHODS(InOutTensorSourceImpl);
};

struct ShapeDimSource {
  InOutTensorSource tensor_source;
  int dim_axis;

  bool operator==(const ShapeDimSource& other) const {
    return this->tensor_source == other.tensor_source &&
           this->dim_axis == other.dim_axis;
  }
};

struct DataDimSource {
  InOutTensorSource tensor_source;
  int dim_axis;

  bool operator==(const DataDimSource& other) const {
    return this->tensor_source == other.tensor_source &&
           this->dim_axis == other.dim_axis;
  }
};

using DimSourceImpl = std::variant<ShapeDimSource, DataDimSource>;
struct DimSource : public DimSourceImpl {
  using DimSourceImpl::DimSourceImpl;
  ADT_DEFINE_VARIANT_METHODS(DimSourceImpl);
};

template <typename BirNode /* backend ir node*/>
struct ArgSourceCtxImpl {
  std::vector<std::pair<BirNode, InTensorSource>> input_and_tensor_source_pairs;
  std::vector<std::pair<BirNode, OutTensorSource>>
      output_and_tensor_source_pairs;
  std::vector<std::pair<symbol::DimExpr, DimSource>>
      dim_expr_and_dim_source_pairs;
  std::unordered_map<symbol::DimExpr, DimSource> dim_expr2dim_source;

  std::optional<const InTensorSource*> GetInputTensorSource(
      const BirNode& node) const {
    for (const auto& [k, v] : this->input_and_tensor_source_pairs) {
      if (k == node) {
        return &v;
      }
    }
    return std::nullopt;
  }

  std::optional<const OutTensorSource*> GetOutputTensorSource(
      const BirNode& node) const {
    for (const auto& [k, v] : this->output_and_tensor_source_pairs) {
      if (k == node) {
        return &v;
      }
    }
    return std::nullopt;
  }

  std::optional<const DimSource*> GetDimExprSource(
      const symbol::DimExpr& dim_expr) const {
    const auto& iter = this->dim_expr2dim_source.find(dim_expr);
    if (iter == this->dim_expr2dim_source.end()) {
      return std::nullopt;
    }
    return &iter->second;
  }

  bool HasDirectOrIndirectDimExprSource(const symbol::DimExpr& dim_expr) const {
    if (GetDimExprSource(dim_expr).has_value()) {
      return true;
    }
    return dim_expr.Match([&](const auto& impl) {
      return HasDirectOrIndirectDimExprSourceImpl(impl);
    });
  }

 private:
  bool HasDirectOrIndirectDimExprSourceImpl(int64_t) const { return true; }

  bool HasDirectOrIndirectDimExprSourceImpl(const std::string& dim_expr) const {
    return GetDimExprSource(dim_expr).has_value();
  }

  using Negative = symbol::Negative<symbol::DimExpr>;
  bool HasDirectOrIndirectDimExprSourceImpl(const Negative& dim_expr) const {
    return HasDirectOrIndirectUnaryDimExprSource(dim_expr);
  }

  template <typename T>
  bool HasDirectOrIndirectUnaryDimExprSource(const T& dim_expr) const {
    const auto& [operand] = *dim_expr;
    return HasDirectOrIndirectDimExprSource(operand);
  }

  using Add = symbol::Add<symbol::DimExpr>;
  bool HasDirectOrIndirectDimExprSourceImpl(const Add& dim_expr) const {
    return HasDirectOrIndirectVariadicDimExprSource(dim_expr);
  }

  using Mul = symbol::Mul<symbol::DimExpr>;
  bool HasDirectOrIndirectDimExprSourceImpl(const Mul& dim_expr) const {
    return HasDirectOrIndirectVariadicDimExprSource(dim_expr);
  }

  using Div = symbol::Div<symbol::DimExpr>;
  bool HasDirectOrIndirectDimExprSourceImpl(const Div& dim_expr) const {
    const auto& operand_lhs = (*dim_expr).lhs;
    const auto& operand_rhs = (*dim_expr).rhs;
    for (const auto& operand : {operand_lhs, operand_rhs}) {
      if (!HasDirectOrIndirectDimExprSource(operand)) {
        return false;
      }
    }
    return true;
  }

  using Max = symbol::Max<symbol::DimExpr>;
  bool HasDirectOrIndirectDimExprSourceImpl(const Max& dim_expr) const {
    return HasDirectOrIndirectVariadicDimExprSource(dim_expr);
  }

  using Min = symbol::Min<symbol::DimExpr>;
  bool HasDirectOrIndirectDimExprSourceImpl(const Min& dim_expr) const {
    return HasDirectOrIndirectVariadicDimExprSource(dim_expr);
  }

  using Broadcast = symbol::Broadcast<symbol::DimExpr>;
  bool HasDirectOrIndirectDimExprSourceImpl(const Broadcast& dim_expr) const {
    return HasDirectOrIndirectVariadicDimExprSource(dim_expr);
  }

  template <typename T>
  bool HasDirectOrIndirectVariadicDimExprSource(const T& dim_expr) const {
    const auto& [operands] = dim_expr;
    for (const auto& operand : *operands) {
      if (!HasDirectOrIndirectDimExprSource(operand)) {
        return false;
      }
    }
    return true;
  }
};

template <typename BirNode /* backend ir node*/>
ADT_DEFINE_RC(ArgSourceCtx, ArgSourceCtxImpl<BirNode>);

}  // namespace ap::code_gen
