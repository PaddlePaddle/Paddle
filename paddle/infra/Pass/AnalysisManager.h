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

/// The design is mainly from MLIR project.

// TODO(wilber): Add implmentation.

#pragma once

#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeName.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/TypeID.h"

#include "Pass/PassInstrumentation.h"

#include "utils/STLExtras.h"

namespace infra {
class AnalysisManager;
class PassInstrumentor;

namespace detail {

/// A utility class to reprensent the analyses that are kwnown to be preserved.
class PreservedAnalyses {
  struct AllAnalysesType {};

 public:
  /// Mark all analyses as preserved.
  void PreserveAll() {
    preserved_ids_.insert(mlir::TypeID::get<AllAnalysesType>());
  }

  bool IsAll() const {
    return preserved_ids_.count(mlir::TypeID::get<AllAnalysesType>());
  }

  bool IsNone() const { return preserved_ids_.empty(); }

  template <typename AnalysisT>
  void Preserve() {
    Preserve(mlir::TypeID::get<AnalysisT>());
  }

  template <typename AnalysisT, typename AnalysisT2, typename... OtherAnalysesT>
  void Preserve() {
    Preserve<AnalysisT>();
    Preserve<AnalysisT2, OtherAnalysesT...>();
  }

  void Preserve(mlir::TypeID id) { preserved_ids_.insert(id); }

  template <typename AnalysisT>
  bool IsPreserved() const {
    return IsPreserved(mlir::TypeID::get<AnalysisT>());
  }

  bool IsPreserved(mlir::TypeID id) const { return preserved_ids_.count(id); }

 private:
  template <typename AnalysisT>
  void Unpreserve() {
    preserved_ids_.erase(mlir::TypeID::get<AnalysisT>());
  }

  template <typename>
  friend struct AnalysisModel;

  llvm::SmallPtrSet<mlir::TypeID, 2> preserved_ids_;
};

namespace analysis_impl {

/// Trait to check if T provides a static `IsInvalidated` method.
template <typename T, typename... Args>
using has_is_invalidated = decltype(std::declval<T&>().IsInvalidated(
    std::declval<const PreservedAnalyses&>()));

/// Implementation of `IsInvalidated` if the analysis provides a definition.
template <typename AnalysisT>
std::enable_if_t<is_detected<has_is_invalidated, AnalysisT>::value, bool>
IsInvalidated(AnalysisT& analysis, const PreservedAnalyses& pa) {  // NOLINT
  return analysis.IsInvalidated(pa);
}

/// Default implementation of `IsInvalidated`.
template <typename AnalysisT>
std::enable_if_t<!is_detected<has_is_invalidated, AnalysisT>::value, bool>
IsInvalidated(AnalysisT& analysis, const PreservedAnalyses& pa) {  // NOLINT
  return !pa.IsPreserved<AnalysisT>();
}
}  // namespace analysis_impl

/// Abstract base class representing an analysis.
struct AnalysisConcept {
  virtual ~AnalysisConcept() = default;

  // A hook used to query analyses for invalidation.
  virtual bool Invalidate(PreservedAnalyses& pa) = 0;  // NOLINT
};

template <typename AnalysisT>
struct AnalysisModel : public AnalysisConcept {
  template <typename... Args>
  explicit AnalysisModel(Args&&... args)
      : analysis(std::forward<Args>(args)...) {}

  bool Invalidate(PreservedAnalyses& pa) final {
    bool result = analysis_impl::IsInvalidated(analysis, pa);
    if (result) pa.Unpreserve<AnalysisT>();
    return result;
  }

  AnalysisT analysis;
};

/// This class represents a cache of analyses for a single operation.
/// All computation, caching and invalidation of analyses takes place here.
class AnalysisMap {
 public:
  explicit AnalysisMap(mlir::Operation* ir) : ir_(ir) {}

  template <typename AnalysisT>
  AnalysisT& GetAnalysis(PassInstrumentor* pi, AnalysisManager& am) {
    return GetAnalysisImpl<AnalysisT, mlir::Operation*>(pi, ir_, am);
  }

  template <typename AnalysisT, typename OpT>
  std::enable_if_t<
      std::is_constructible<AnalysisT, OpT>::value ||
          std::is_constructible<AnalysisT, OpT, AnalysisManager&>::value,
      AnalysisT&>
  GetAnalysis(PassInstrumentor* pi, AnalysisManager& am) {  // NOLINT
    return GetAnalysisImpl<AnalysisT, OpT>(pi, llvm::cast<OpT>(ir_), am);
  }

  template <typename AnalysisT>
  std::optional<std::reference_wrapper<AnalysisT>> GetCachedAnalysis() const {
    auto res = analyses_.find(mlir::TypeID::get<AnalysisT>());
    if (res == analyses_.end()) return std::nullopt;
    return {static_cast<AnalysisModel<AnalysisT>&>(*res->second).analysis};
  }

  mlir::Operation* getOperation() const { return ir_; }

  void Clear() { analyses_.clear(); }

  /// Invalidate any cached analyses based upon the given set of preserved
  void Invalidate(const PreservedAnalyses& pa) {
    PreservedAnalyses pa_copy(pa);

    // Remove any analyses that were invalidaed.
    // As using MapVector, order of insertion is preserved and
    // dependencies always go before users, so need only one iteration.
    analyses_.remove_if(
        [&](auto& val) { return val.second->Invalidate(pa_copy); });
  }

 private:
  template <typename AnalysisT>
  static llvm::StringRef GetAnalysisName() {
    llvm::StringRef name = llvm::getTypeName<AnalysisT>();
    if (!name.consume_front("infra::"))
      name.consume_front("(anonymous namespace)::");
    return name;
  }

  template <typename AnalysisT, typename OpT>
  AnalysisT& GetAnalysisImpl(PassInstrumentor* pi,
                             OpT op,
                             AnalysisManager& am) {  // NOLINT
    mlir::TypeID id = mlir::TypeID::get<AnalysisT>();
    auto it = analyses_.find(id);
    if (it == analyses_.end()) {
      if (pi) {
        pi->RunBeforeAnalysis(GetAnalysisName<AnalysisT>().str(), id, ir_);
      }

      bool was_inserted;
      std::tie(it, was_inserted) =
          analyses_.insert({id, ConstructAnalysis<AnalysisT>(am, op)});
      assert(was_inserted);

      if (pi) {
        pi->RunAfterAnalysis(GetAnalysisName<AnalysisT>().str(), id, ir_);
      }
    }

    return static_cast<AnalysisModel<AnalysisT>&>(*it->second).analysis;
  }

  /// Construct analysis using two arguments constructor (OpT,
  /// AnalysisManager&).
  template <
      typename AnalysisT,
      typename OpT,
      std::enable_if_t<
          std::is_constructible<AnalysisT, OpT, AnalysisManager&>::value>* =
          nullptr>
  static auto ConstructAnalysis(AnalysisManager& am, OpT op) {  // NOLINT
    return std::make_unique<AnalysisModel<AnalysisT>>(op, am);
  }

  /// Construct analysis using single argument constructor (OpT)
  template <
      typename AnalysisT,
      typename OpT,
      std::enable_if_t<
          !std::is_constructible<AnalysisT, OpT, AnalysisManager&>::value>* =
          nullptr>
  static auto ConstructAnalysis(AnalysisManager&, OpT op) {
    return std::make_unique<AnalysisModel<AnalysisT>>(op);
  }

 private:
  mlir::Operation* ir_;
  // std::map<mlir::TypeID, std::unique_ptr<AnalysisConcept>> analyses_;
  llvm::MapVector<mlir::TypeID, std::unique_ptr<AnalysisConcept>> analyses_;
};

}  // namespace detail

/// This class is intended to be passed around by value, and can not be
/// constructed direcyly.
class AnalysisManager {
 public:
  using PreservedAnalyses = detail::PreservedAnalyses;

  template <typename AnalysisT>
  AnalysisT& GetAnalysis() {
    return analyses_->GetAnalysis<AnalysisT>(GetPassInstrumentor(), *this);
  }

  template <typename AnalysisT, typename OpT>
  AnalysisT& GetAnalysis() {
    return analyses_->GetAnalysis<AnalysisT, OpT>(GetPassInstrumentor(), *this);
  }

  template <typename AnalysisT>
  std::optional<std::reference_wrapper<AnalysisT>> GetCachedAnalysis() const {
    return analyses_->GetCachedAnalysis<AnalysisT>();
  }

  void Invalidate(const PreservedAnalyses& pa) {
    if (pa.IsAll()) return;

    // Invalidate the analyses for the current operation directly.
    analyses_->Invalidate(pa);
  }

  void clear() { analyses_->Clear(); }

  PassInstrumentor* GetPassInstrumentor() const { return instrumentor_; }

  mlir::Operation* GetOperation() { return analyses_->getOperation(); }

 private:
  AnalysisManager(detail::AnalysisMap* impl, PassInstrumentor* pi)
      : analyses_(impl), instrumentor_(pi) {}
  friend class AnalysisManagerHolder;

 private:
  detail::AnalysisMap* analyses_;
  PassInstrumentor* instrumentor_;
};

/// A manager class for the container operation. This class hold the
/// memory for the analyses. AnalysisManager just hold the ref to the
/// analyses.
class AnalysisManagerHolder {
 public:
  AnalysisManagerHolder(mlir::Operation* op, PassInstrumentor* pi)
      : analyses_(op), pi_(pi) {}
  AnalysisManagerHolder(const AnalysisManagerHolder&) = delete;
  AnalysisManagerHolder& operator=(const AnalysisManagerHolder&) = delete;

  /// Returns an analysis manager for the current container op.
  operator AnalysisManager() { return AnalysisManager(&analyses_, pi_); }

 private:
  detail::AnalysisMap analyses_;
  PassInstrumentor* pi_;
};

}  // namespace infra
