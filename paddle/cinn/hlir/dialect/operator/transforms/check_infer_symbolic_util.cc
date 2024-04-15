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

#include <memory>
#include <optional>
#include <functional>
#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/fluid/pir/transforms/shape_optimization_pass.h"
#include "paddle/common/flags.h"

COMMON_DECLARE_bool(check_infer_symbolic);

namespace cinn {
namespace dialect {
namespace ir {

namespace {

OptDimExprs4ValueT MakeOptDimExprs4Value(
    pir::Program* program,
    const PassManagerCreater& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  pass_manager->Run(program);
  const auto* shape_analysis =
      &pir::ShapeAnalysisManager::Binding().Get(program);
  return [shape_analysis](pir::Value value, const pir::Block*)
    -> std::optional<const symbol::ShapeOrDataDimExprs*> {
    if (!shape_analysis->HasShapeOrDataForValue(value)) return std::nullopt;
    return &shape_analysis->GetShapeOrDataForValue(value);
  };
}

template <typename DoEachT>
void WalkLeafOp(pir::Program* program, const DoEachT& DoEach) {
  program->module_op().Walk([&](pir::Operation* op) {
    if (op->num_regions() != 0) return;
    DoEach(op);
  });
}

struct ShapeSignatureGenerator {
  pir::Operation* op;
  OptDimExprs4ValueT GraphDimExprs4Value;

  using SymbolBindings = std::map<std::string, int64_t>;

  using ShapeAnalysisPtr = std::shared_ptr<pir::ShapeConstraintIRAnalysis>;
  using ShapeList = std::vector<std::vector<int64_t>>;
  using DoEachShapeSignatureT =
    std::function<void(const ShapeList& inputs, const ShapeList& outputs)>;
  template <typename DoEachT>
  void Generate(const DoEachShapeSignatureT& DoEachShapeSignature) {
    auto op_shape_analysis = MakeOpShapeAnalysis(op, GraphDimExprs4Value);
    VisitInputSymbolBinding(op_shape_analysis, [&](const auto& bindings){
      const auto input_shapes = GetInputShapes(op_shape_analysis, bindings);
      const auto output_shapes = GetOutputShapes(op_shape_analysis, bindings);
      DoEachShapeSignature(input_shapes, output_shapes);
    });
  }

  ShapeList GetInputShapes(
      const ShapeAnalysisPtr& op_shape_analysis,
      const SymbolBindings& bindings) {
    TODO("jiawenxuan");
  }

  ShapeList GetOutputShapes(
      const ShapeAnalysisPtr& op_shape_analysis,
      const SymbolBindings& bindings) {
    TODO("jiawenxuan");
  }

  struct CstrEqSymbolNames {
    std::vector<std::string> symbol_names;
  };

  struct CstrBroadcastableSymbolNames {
    std::vector<std::string> symbol_names;
  };

  using ConstrainedSymbolNames =
      std::variant<CstrEqSymbolNames, CstrBroadcastableSymbolNames>;

  using ConstrainedSymbolNamesList = std::vector<ConstrainedSymbolNames>;


  using ConstrainedSymbolNamesAndDimBindingList = 
    std::vector<std::pair<ConstrainedSymbolNames, int64_t>>;

  using DoEachSymbolBindingT =
    std::function<void(const SymbolBindings&)>;
  void VisitInputSymbolBinding(
      const ShapeAnalysisPtr& op_shape_analysis,
      const DoEachSymbolBindingT& DoEachSymbolBinding) {

    ConstrainedSymbolNamesList constrained_sym_list =
      GetConstrainedSymbolNamesList(op_shape_analysis);
    VisitSymbolsBindings(constrained_sym_list, [&](const auto& syms_and_dims){
      VisitSymbolBindings(syms_and_dims, {}, DoEachSymbolBinding);
    });
  }

  void VisitSymbolBindings(
      const ConstrainedSymbolNamesAndDimBindingList& syms_and_dims,
      const SymbolBindings& collected,
      const DoEachSymbolBindingT& DoEachSymbolBinding) {
    if (syms_and_dims.empty()) return DoEachSymbolBinding(collected);
    ConstrainedSymbolNamesAndDimBindingList remainder{
      std::next(syms_and_dims.begin()), syms_and_dims.end()
    };
    const auto* first = &syms_and_dims.begin();
    VisitConstrainedSymbolBindings(*first, [&](const auto& cur_bindings){
      if (HasConflict(collected, cur_bindings)) return;
      cur_bindings.insert(collected.begin(), collected.end());
      VisitSymbolBindings(remainder, cur_bindings, DoEachSymbolBinding);
    });
  }

  bool HasConflict(
      const SymbolBindings& lhs,
      const SymbolBindings& rhs) {
    for (const auto& [sym_name, dim] : lhs) {
      const auto& iter = rhs.find(sym_name);
      if (iter == rhs.end()) continue;
      if (iter->second != dim) return true;
    }
    return false;
  }

  template <typename DoEachT>
  void VisitConstrainedSymbolBindings(
      std::pair<ConstrainedSymbolNames, int64_t>& syms_and_dim,
      const DoEachT& DoEach) {
    const auto& [syms, dim] = syms_and_dim;
    return std::visit([&](const auto& impl) {
      return VisitConstrainedSymbolBindingsImpl(impl, dim, DoEach);
    }, syms);
  }

  template <typename DoEachT>
  void VisitConstrainedSymbolBindingsImpl(
      const CstrEqSymbolNames& syms,
      int64_t dim,
      const DoEachT& DoEach) {
    SymbolBindings bindings;
    for (const auto& sym_name : syms.symbol_names) {
      bindings[sym_name] = dim;
    }
    DoEach(bindings);
  }

  template <typename DoEachT>
  void VisitConstrainedSymbolBindingsImpl(
      const CstrBroadcastableSymbolNames& syms,
      int64_t dim,
      const DoEachT& DoEach) {
    VisitEachSubSet(syms.symbol_names.size(), {}, [&](const auto& flags) {
      SymbolBindings bindings;
      for (int i = 0; i < syms.symbol_names.size(); ++i) {
        bindings[syms.symbol_names.at(i)] = (flags.at(i) ? dim : 1);
      }
      DoEach(bindings);
    });
  }

  using IsSubset = int;

  template <typename DoEachT>
  void VisitEachSubSet(
      int set_size,
      const std::vector<IsSubset>& is_subset_flags,
      const DoEachT& DoEach) {
    if (set_size <= 0) return DoEach(is_subset_flags);

    const auto& RecusiveVisit = [&](bool is_subset) {
      std::vector<IsSubset> current_is_subset_flags(is_subset_flags);
      current_is_subset_flags.push_back(int(is_subset));
      VisitEachSubSet(set_size - 1, current_is_subset_flags, DoEach);
    };
    RecusiveVisit(true);
    RecusiveVisit(false);
  }

  ConstrainedSymbolNamesList GetConstrainedSymbolNamesList(
      const ShapeAnalysisPtr& op_shape_analysis) {
    TODO("jiawenxuan");
  }

  using ConstrainedSymbolNamesAndDimBindingsList = 
    std::vector<std::pair<ConstrainedSymbolNames, std::vector<int64_t>>>;

  template <typename DoEachT>
  void VisitSymbolsBindings(
      const ConstrainedSymbolNamesList& constrained_sym_list,
      const DoEachT& DoEach) {
    if (constrained_sym_list.empty()) return;
    ConstrainedSymbolNamesAndDimBindingsList names_and_dims
        = GetConstrainedSymbolNamesAndDimBindingsList(constrained_sym_list);
    VisitCombinedSymbolsBindings(names_and_dims, {}, DoEach);
  }

  ConstrainedSymbolNamesAndDimBindingsList
  GetConstrainedSymbolNamesAndDimBindingsList(
      const ConstrainedSymbolNamesList& constrained_sym_list) {
    TODO("jiayunxuan");
  }

  template <typename DoEachT>
  void VisitCombinedSymbolsBindings(
      const ConstrainedSymbolNamesAndDimBindingsList& names_and_dims,
      const ConstrainedSymbolNamesAndDimBindingList& names_and_dim
      const DoEachT& DoEach) {
    if (names_and_dims.empty()) return DoEach(names_and_dim);
    ConstrainedSymbolNamesAndDimBindingsList remainder{
      std::next(names_and_dims.begin()), names_and_dims.end()
    };
    const auto* first = &names_and_dims.at(0);
    auto cur_names_and_dim = 
          ConstrainedSymbolNamesAndDimBindingList(names_and_dim);
      cur_names_and_dim.push_back(std::pair(first->first, 0));
    for (int64_t dim_binding : first->second) {
      cur_names_and_dim.back().second = dim_binding;
      VisitCombinedSymbolsBindings(remainder, cur_names_and_dim);
    }
  }

};

void CheckByInferMeta(
    pir::Operation* op,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<int64_t>>& output_shapes) {
  TODO("jiawenxuan");
}

void CheckOpDimExprConstraints(
    pir::Operation* op,
    const OptDimExprs4ValueT& GraphDimExprs4Value) {
  ShapeSignatureGenerator generator(op, GraphDimExprs4Value);
  generator.Generate([&](const auto& input_shapes, const auto& output_shapes) {
    CheckByInferMeta(op, input_shapes, output_shapes);
  });
}

void CheckProgramDimExprConstraints(
    pir::Program* program,
    const OptDimExprs4ValueT& GraphDimExprs4Value) {
  WalkLeafOp(program, [&](pir::Operation* op){
    CheckOpDimExprConstraints(op, GraphDimExprs4Value);
  });
}

}

void CheckInferSymbolicIfNeed(pir::Program* program,
                              const PassManagerCreater& CreatePassManager) {
  if (!FLAGS_check_infer_symbolic) return;
  const auto& GraphDimExprs4Value =
    MakeOptDimExprs4Value(program, CreatePassManager);
  CheckProgramDimExprConstraints(program, GraphDimExprs4Value);
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(CreateCheckInferSymbolicPass(GraphDimExprs4Value));
  pass_manager->AddPass(CreateSplitGenerateShapeIntoShapeOpsPass());
  pass_manager->Run(program);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
