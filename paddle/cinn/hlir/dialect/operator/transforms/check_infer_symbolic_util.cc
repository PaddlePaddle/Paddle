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

#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_util.h"

#include <functional>
#include <memory>
#include <optional>
#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/local_infer_symbolic_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"

COMMON_DECLARE_bool(check_infer_symbolic);
PD_DECLARE_bool(prim_all);

namespace cinn {
namespace dialect {
namespace ir {

namespace {

std::ostream& operator<<(std::ostream& stream,
                         const std::vector<std::vector<int64_t>>& shapes) {
  for (auto shape : shapes) {
    stream << "[";
    for (auto dim : shape) {
      stream << dim << " ";
    }
    stream << "] ";
  }
  return stream;
}

DimExprs4ValueT MakeDimExprs4Value(
    pir::Program* program, const PassManagerCreater& CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  pass_manager->Run(program);
  auto* shape_analysis = &pir::ShapeAnalysisManager::Instance().Get(program);
  return
      [shape_analysis](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
        // TODO(Hongqing-work): define a default empty ShapeOrDataDimExprss
        if (!value) {
          static symbol::ShapeOrDataDimExprs empty{
              symbol::TensorShapeOrDataDimExprs{}};
          return empty;
        }
        return shape_analysis->GetShapeOrDataForValue(value);
      };
}

template <typename DoEachT>
void VisitEachOp(const pir::ModuleOp& op, const DoEachT& DoEach) {
  for (uint32_t i = 0; i < op->num_regions(); i++) {
    for (pir::Block& block : op->region(i)) {
      for (pir::Operation& sub_op : block) {
        DoEach(&sub_op);
      }
    }
  }
}

template <typename DoEachT>
void WalkLeafOp(pir::Program* program, const DoEachT& DoEach) {
  auto module_op = program->module_op();
  VisitEachOp(module_op, [&](pir::Operation* op) {
    if (op->num_regions() != 0) return;
    DoEach(op);
  });
}

struct ShapeSignatureGenerator {
  pir::Operation* op;
  DimExprs4ValueT GraphDimExprs4Value;

  ShapeSignatureGenerator(pir::Operation* op_,
                          DimExprs4ValueT GraphDimExprs4Value_)
      : op(op_), GraphDimExprs4Value(GraphDimExprs4Value_) {}

  using SymbolBindings = std::map<std::string, int64_t>;
  using ShapeAnalysisPtr = std::shared_ptr<pir::ShapeConstraintIRAnalysis>;
  using Shape = std::vector<int64_t>;
  using ShapeList = std::vector<Shape>;
  using DoEachShapeSignatureT =
      std::function<void(const ShapeList& inputs, const ShapeList& outputs)>;

  void Generate(const DoEachShapeSignatureT& DoEachShapeSignature) {
    auto op_shape_analysis = MakeOpShapeAnalysis(op, GraphDimExprs4Value);
    if (op_shape_analysis.use_count() == 0) return;
    VisitInputSymbolBinding(
        op_shape_analysis, [&](const SymbolBindings& bindings) {
          const auto& substitute_pattern =
              GetSubstitutePattern(bindings, op_shape_analysis);
          const auto input_shapes =
              GetInputShapes(*op, op_shape_analysis, substitute_pattern);
          const auto output_shapes =
              GetOutputShapes(*op, op_shape_analysis, substitute_pattern);
          DoEachShapeSignature(input_shapes, output_shapes);
        });
  }

  std::unordered_map<symbol::DimExpr, symbol::DimExpr> GetSubstitutePattern(
      const SymbolBindings& bindings,
      std::shared_ptr<pir::ShapeConstraintIRAnalysis> op_shape_analysis) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<int> distr(0, 1000);
    std::unordered_map<symbol::DimExpr, symbol::DimExpr> substitute_pattern(
        bindings.begin(), bindings.end());
    int64_t symbol_index = op_shape_analysis->GetSymbolIndex();
    for (int i = 0; i <= symbol_index; i++) {
      symbol::DimExpr dim_expr("S" + std::to_string(i));
      if (substitute_pattern.count(dim_expr) <= 0) {
        substitute_pattern[dim_expr] = symbol::DimExpr(distr(eng));
      }
    }
    return substitute_pattern;
  }

  Shape ConvertSymbolToDim(
      symbol::ShapeOrDataDimExprs shape_or_data,
      const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
          substitute_pattern) {
    Shape dim_shape;
    const auto& const_shape_or_data =
        symbol::SubstituteShapeOrData(shape_or_data, substitute_pattern);
    for (const auto& symbolic_shape : const_shape_or_data.shape()) {
      const auto& const_symbolic_shape =
          symbol::SimplifyDimExpr(symbolic_shape);
      CHECK(const_symbolic_shape.isa<std::int64_t>());
      dim_shape.push_back(symbolic_shape.Get<std::int64_t>());
    }
    return dim_shape;
  }

  ShapeList GetInputShapes(
      const pir::Operation& op,
      const ShapeAnalysisPtr& op_shape_analysis,
      const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
          substitute_pattern) {
    ShapeList list;
    for (std::size_t i = 0; i < op.num_operands(); ++i) {
      const symbol::ShapeOrDataDimExprs& shape_or_data =
          op_shape_analysis->GetShapeOrDataForValue(op.operand_source(i));
      const auto& dim_shape =
          ConvertSymbolToDim(shape_or_data, substitute_pattern);
      list.emplace_back(dim_shape);
    }
    return list;
  }

  ShapeList GetOutputShapes(
      const pir::Operation& op,
      const ShapeAnalysisPtr& op_shape_analysis,
      const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
          substitute_pattern) {
    ShapeList list;
    for (std::size_t i = 0; i < op.num_results(); ++i) {
      const symbol::ShapeOrDataDimExprs& shape_or_data =
          op_shape_analysis->GetShapeOrDataForValue(op.result(i));
      const auto& shape = ConvertSymbolToDim(shape_or_data, substitute_pattern);
      list.emplace_back(shape);
    }
    return list;
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

  using DoEachSymbolBindingT = std::function<void(const SymbolBindings&)>;
  void VisitInputSymbolBinding(
      const ShapeAnalysisPtr& op_shape_analysis,
      const DoEachSymbolBindingT& DoEachSymbolBinding) {
    ConstrainedSymbolNamesList constrained_sym_list =
        GetConstrainedSymbolNamesList(op_shape_analysis);
    for (ConstrainedSymbolNames constraint : constrained_sym_list) {
      if (std::holds_alternative<CstrEqSymbolNames>(constraint)) {
        auto eq = std::get<CstrEqSymbolNames>(constraint);
      }
      if (std::holds_alternative<CstrBroadcastableSymbolNames>(constraint)) {
        auto eq = std::get<CstrBroadcastableSymbolNames>(constraint);
      }
    }

    VisitSymbolsBindings(constrained_sym_list, [&](const auto& syms_and_dims) {
      VisitSymbolBindings(syms_and_dims, {}, DoEachSymbolBinding);
    });
  }

  void VisitSymbolBindings(
      const ConstrainedSymbolNamesAndDimBindingList& syms_and_dims,
      const SymbolBindings& collected,
      const DoEachSymbolBindingT& DoEachSymbolBinding) {
    if (syms_and_dims.empty()) return DoEachSymbolBinding(collected);
    ConstrainedSymbolNamesAndDimBindingList remainder{
        std::next(syms_and_dims.begin()), syms_and_dims.end()};
    const auto& first = syms_and_dims.at(0);
    VisitConstrainedSymbolBindings(first, [&](auto& cur_bindings) {
      if (HasConflict(collected, cur_bindings)) return;
      cur_bindings.insert(collected.begin(), collected.end());
      VisitSymbolBindings(remainder, cur_bindings, DoEachSymbolBinding);
    });
  }

  bool HasConflict(const SymbolBindings& lhs, const SymbolBindings& rhs) {
    for (const auto& [sym_name, dim] : lhs) {
      const auto& iter = rhs.find(sym_name);
      if (iter == rhs.end()) continue;
      if (iter->second != dim) return true;
    }
    return false;
  }

  template <typename DoEachT>
  void VisitConstrainedSymbolBindings(
      const std::pair<ConstrainedSymbolNames, int64_t>& syms_and_dim,
      const DoEachT& DoEach) {
    const auto& [syms, dim] = syms_and_dim;
    return std::visit(
        [&](const auto& impl) {
          return VisitConstrainedSymbolBindingsImpl(impl, dim, DoEach);
        },
        syms);
  }

  template <typename DoEachT>
  void VisitConstrainedSymbolBindingsImpl(const CstrEqSymbolNames& syms,
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
  void VisitEachSubSet(int set_size,
                       const std::vector<IsSubset>& is_subset_flags,
                       const DoEachT& DoEach) {
    if (set_size <= 0) return DoEach(is_subset_flags);

    const auto& RecusiveVisit = [&](bool is_subset) {
      std::vector<IsSubset> current_is_subset_flags(is_subset_flags);
      current_is_subset_flags.push_back(static_cast<int>(is_subset));
      VisitEachSubSet(set_size - 1, current_is_subset_flags, DoEach);
    };
    RecusiveVisit(true);
    RecusiveVisit(false);
  }

  ConstrainedSymbolNamesList GetConstrainedSymbolNamesList(
      const ShapeAnalysisPtr& op_shape_analysis) {
    ConstrainedSymbolNamesList cstr_list;
    auto* cstr_manager = op_shape_analysis->GetConstraintsManager();
    cstr_manager->VisitEqualClusters([&](auto clusters) {
      CstrEqSymbolNames equals;
      for (const symbol::DimExpr& dim_expr : clusters) {
        if (dim_expr.isa<std::string>())
          equals.symbol_names.push_back(dim_expr.Get<std::string>());
      }
      if (!equals.symbol_names.empty()) cstr_list.emplace_back(equals);
    });

    cstr_manager->BroadcastableConstraintsVisitor([&](auto it) {
      if (it->data->lhs.template isa<std::string>() &&
          it->data->rhs.template isa<std::string>()) {
        CstrBroadcastableSymbolNames bcables;
        bcables.symbol_names.push_back(
            it->data->lhs.template Get<std::string>());
        bcables.symbol_names.push_back(
            it->data->rhs.template Get<std::string>());
        cstr_list.emplace_back(bcables);
      }
    });

    return cstr_list;
  }

  using ConstrainedSymbolNamesAndDimBindingsList =
      std::vector<std::pair<ConstrainedSymbolNames, std::vector<int64_t>>>;

  template <typename DoEachT>
  void VisitSymbolsBindings(
      const ConstrainedSymbolNamesList& constrained_sym_list,
      const DoEachT& DoEach) {
    if (constrained_sym_list.empty()) return;
    ConstrainedSymbolNamesAndDimBindingsList names_and_dims =
        GetConstrainedSymbolNamesAndDimBindingsList(constrained_sym_list);
    VisitCombinedSymbolsBindings(names_and_dims, {}, DoEach);
  }

  ConstrainedSymbolNamesAndDimBindingsList
  GetConstrainedSymbolNamesAndDimBindingsList(
      const ConstrainedSymbolNamesList& constrained_sym_list) {
    static const std::vector<int64_t> table{1, 2, 3, 4};

    ConstrainedSymbolNamesAndDimBindingsList list;
    for (const auto& cstr : constrained_sym_list) {
      list.push_back(std::pair(cstr, table));
    }
    return list;
  }

  template <typename DoEachT>
  void VisitCombinedSymbolsBindings(
      const ConstrainedSymbolNamesAndDimBindingsList& names_and_dims,
      const ConstrainedSymbolNamesAndDimBindingList& names_and_dim,
      const DoEachT& DoEach) {
    if (names_and_dims.empty()) return DoEach(names_and_dim);
    ConstrainedSymbolNamesAndDimBindingsList remainder{
        std::next(names_and_dims.begin()), names_and_dims.end()};
    const auto* first = &names_and_dims.at(0);
    auto cur_names_and_dim =
        ConstrainedSymbolNamesAndDimBindingList(names_and_dim);
    cur_names_and_dim.push_back(std::pair(first->first, 0));
    for (int64_t dim_binding : first->second) {
      cur_names_and_dim.back().second = dim_binding;
      VisitCombinedSymbolsBindings(remainder, cur_names_and_dim, DoEach);
    }
  }
};

void DoInferMeta(const std::vector<std::vector<int64_t>>& input_shapes,
                 pir::Builder* builder,
                 pir::Operation* op,
                 std::vector<paddle::dialect::EmptyOp>* empty_op_list,
                 std::vector<std::vector<int64_t>>* infer_meta_result) {
  std::vector<pir::Value> input_values;
  for (int i = 0; i < input_shapes.size(); i++) {
    paddle::dialect::EmptyOp empty_op =
        builder->Build<paddle::dialect::EmptyOp>(input_shapes[i]);
    empty_op_list->push_back(empty_op);
    input_values.push_back(empty_op.out());
  }

  pir::AttributeMap attribute_map = op->attributes();
  paddle::dialect::InferMetaInterface interface =
      op->dyn_cast<paddle::dialect::InferMetaInterface>();
  const auto& types = interface.InferMeta(input_values, &attribute_map);
  for (const auto& type : types) {
    infer_meta_result->push_back(
        common::vectorize(type.dyn_cast<pir::DenseTensorType>().dims()));
  }
}

void EraseEmptyOp(const std::vector<paddle::dialect::EmptyOp>& empty_op_list) {
  for (auto& empty_op : empty_op_list) {
    PADDLE_ENFORCE_EQ(
        empty_op->use_empty(),
        true,
        phi::errors::InvalidArgument("Erase op failed. op(%s) is used, the "
                                     "expectation is that it is not used",
                                     empty_op->name()));
    empty_op->Erase();
  }
}

void CheckByInferMeta(pir::Operation* op,
                      pir::Builder* builder,
                      const std::vector<std::vector<int64_t>>& input_shapes,
                      const std::vector<std::vector<int64_t>>& output_shapes) {
  std::vector<paddle::dialect::EmptyOp> empty_op_list;
  std::vector<std::vector<int64_t>> infer_meta_result;
  DoInferMeta(input_shapes, builder, op, &empty_op_list, &infer_meta_result);

  CHECK(infer_meta_result.size() == output_shapes.size())
      << "check " << op->name() << " constraints error";
  for (int i = 0; i < infer_meta_result.size(); i++) {
    CHECK(infer_meta_result[i].size() == output_shapes[i].size())
        << "check " << op->name() << " constraints error";
    for (int j = 0; j < infer_meta_result[i].size(); j++) {
      if (infer_meta_result[i][j] != -1)
        CHECK(infer_meta_result[i][j] == output_shapes[i][j])
            << "check " << op->name() << " constraints error";
    }
  }
  EraseEmptyOp(empty_op_list);
}

void CheckOpDimExprConstraints(pir::Operation* op,
                               const DimExprs4ValueT& GraphDimExprs4Value) {
  ShapeSignatureGenerator generator(op, GraphDimExprs4Value);
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Builder builder = pir::Builder(ctx, op->GetParent());
  generator.Generate([&](const auto& input_shapes, const auto& output_shapes) {
    CheckByInferMeta(op, &builder, input_shapes, output_shapes);
  });
}

void CheckProgramDimExprConstraints(
    pir::Program* program, const DimExprs4ValueT& GraphDimExprs4Value) {
  WalkLeafOp(program, [&](pir::Operation* op) {
    if (op->isa<pir::ShadowOutputOp>()) return;
    VLOG(4) << "########Check Constraints for : " << op->name()
            << " ################";
    CheckOpDimExprConstraints(op, GraphDimExprs4Value);
  });
}

}  // namespace

void CheckInferSymbolicIfNeed(pir::Program* program,
                              const PassManagerCreater& CreatePassManager) {
  if (!FLAGS_prim_all || !FLAGS_check_infer_symbolic) return;
  const auto& GraphDimExprs4Value =
      MakeDimExprs4Value(program, CreatePassManager);
  CheckProgramDimExprConstraints(program, GraphDimExprs4Value);
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(CreateCheckInferSymbolicPass(GraphDimExprs4Value));
  pass_manager->AddPass(CreateSplitGenerateShapeIntoShapeOpsPass());
  pass_manager->Run(program);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
