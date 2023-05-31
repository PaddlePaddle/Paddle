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

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/ir/core/cast_utils.h"
#include "paddle/ir/core/type_id.h"
#include "paddle/ir/core/type_name.h"
#include "paddle/ir/pass/pass_instrumentation.h"
#include "paddle/ir/pass/utils.h"
#include "paddle/utils/optional.h"

namespace ir {

class Operation;
class AnalysisManager;
class PassInstrumentor;

namespace detail {

/// A utility class to reprensent the analyses that are kwnown to be preserved.
class PreservedAnalyses {
  struct AllAnalysesType {};

 public:
  /// Mark all analyses as preserved.
  void PreserveAll() {
    preserved_ids_.insert(ir::TypeId::get<AllAnalysesType>());
  }

  bool IsAll() const {
    return preserved_ids_.count(ir::TypeId::get<AllAnalysesType>());
  }

  bool IsNone() const { return preserved_ids_.empty(); }

  template <typename AnalysisT>
  void Preserve() {
    Preserve(ir::TypeId::get<AnalysisT>());
  }

  template <typename AnalysisT, typename AnalysisT2, typename... OtherAnalysesT>
  void Preserve() {
    Preserve<AnalysisT>();
    Preserve<AnalysisT2, OtherAnalysesT...>();
  }

  void Preserve(ir::TypeId id) { preserved_ids_.insert(id); }

  template <typename AnalysisT>
  bool IsPreserved() const {
    return IsPreserved(ir::TypeId::get<AnalysisT>());
  }

  bool IsPreserved(ir::TypeId id) const { return preserved_ids_.count(id); }

  template <typename AnalysisT>
  void Unpreserve() {
    preserved_ids_.erase(ir::TypeId::get<AnalysisT>());
  }

 private:
  template <typename>
  friend struct AnalysisModel;

  std::unordered_set<ir::TypeId> preserved_ids_;
};

namespace detail {

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
}  // namespace detail

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
    bool result = detail::IsInvalidated(analysis, pa);
    if (result) pa.Unpreserve<AnalysisT>();
    return result;
  }

  AnalysisT analysis;
};

/// This class represents a cache of analyses for a single operation.
/// All computation, caching and invalidation of analyses takes place here.
class AnalysisMap {
 public:
  explicit AnalysisMap(ir::Operation* ir) : ir_(ir) {}

  template <typename AnalysisT>
  AnalysisT& GetAnalysis(PassInstrumentor* pi, AnalysisManager& am) {
    return GetAnalysisImpl<AnalysisT, ir::Operation*>(pi, ir_, am);
  }

  template <typename AnalysisT, typename OpT>
  std::enable_if_t<
      std::is_constructible<AnalysisT, OpT>::value ||
          std::is_constructible<AnalysisT, OpT, AnalysisManager&>::value,
      AnalysisT&>
  GetAnalysis(PassInstrumentor* pi, AnalysisManager& am) {  // NOLINT
    return GetAnalysisImpl<AnalysisT, OpT>(pi, ir::cast<OpT>(ir_), am);
  }

  template <typename AnalysisT>
  paddle::optional<std::reference_wrapper<AnalysisT>> GetCachedAnalysis()
      const {
    auto res = analyses_.find(ir::TypeId::get<AnalysisT>());
    if (res == analyses_.end()) return paddle::none;
    return {static_cast<AnalysisModel<AnalysisT>&>(*res->second).analysis};
  }

  ir::Operation* getOperation() const { return ir_; }

  void Clear() { analyses_.clear(); }

  /// Invalidate any cached analyses based upon the given set of preserved
  void Invalidate(const PreservedAnalyses& pa) {
    PreservedAnalyses pa_copy(pa);

    // Remove any analyses that were invalidaed.
    // As using MapVector, order of insertion is preserved and
    // dependencies always go before users, so need only one iteration.
    for (auto it = analyses_.begin(); it != analyses_.end();) {
      if (it->second->Invalidate(pa_copy))
        it = analyses_.erase(it);
      else
        ++it;
    }
  }

 private:
  template <typename AnalysisT>
  static std::string GetAnalysisName() {
    std::string name = ir::get_type_name<AnalysisT>();
    auto pos = name.rfind("::");
    if (pos != std::string::npos) {
      name = name.substr(pos + 2);
    }
    return name;
  }

  template <typename AnalysisT, typename OpT>
  AnalysisT& GetAnalysisImpl(PassInstrumentor* pi,
                             OpT op,
                             AnalysisManager& am) {  // NOLINT
    ir::TypeId id = ir::TypeId::get<AnalysisT>();
    auto it = analyses_.find(id);
    if (it == analyses_.end()) {
      if (pi) {
        pi->RunBeforeAnalysis(GetAnalysisName<AnalysisT>(), id, ir_);
      }

      bool was_inserted;
      std::tie(it, was_inserted) =
          analyses_.insert({id, ConstructAnalysis<AnalysisT>(am, op)});
      assert(was_inserted);

      if (pi) {
        pi->RunAfterAnalysis(GetAnalysisName<AnalysisT>(), id, ir_);
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
  ir::Operation* ir_;
  std::unordered_map<ir::TypeId, std::unique_ptr<AnalysisConcept>> analyses_;
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
  paddle::optional<std::reference_wrapper<AnalysisT>> GetCachedAnalysis()
      const {
    return analyses_->GetCachedAnalysis<AnalysisT>();
  }

  void Invalidate(const PreservedAnalyses& pa) {
    if (pa.IsAll()) return;

    // Invalidate the analyses for the current operation directly.
    analyses_->Invalidate(pa);
  }

  void clear() { analyses_->Clear(); }

  PassInstrumentor* GetPassInstrumentor() const { return instrumentor_; }

  ir::Operation* GetOperation() { return analyses_->getOperation(); }

 private:
  AnalysisManager(detail::AnalysisMap* impl, PassInstrumentor* pi)
      : analyses_(impl), instrumentor_(pi) {}

 private:
  detail::AnalysisMap* analyses_;
  PassInstrumentor* instrumentor_;

  // For access constructor.
  friend class AnalysisManagerHolder;
};

/// A manager class for the container operation. This class hold the
/// memory for the analyses. AnalysisManager just hold the ref to the
/// analyses.
class AnalysisManagerHolder {
 public:
  AnalysisManagerHolder(ir::Operation* op, PassInstrumentor* pi)
      : analyses_(op), pi_(pi) {}
  AnalysisManagerHolder(const AnalysisManagerHolder&) = delete;
  AnalysisManagerHolder& operator=(const AnalysisManagerHolder&) = delete;

  /// Returns an analysis manager for the current container op.
  operator AnalysisManager() { return AnalysisManager(&analyses_, pi_); }

 private:
  detail::AnalysisMap analyses_;
  PassInstrumentor* pi_;
};

}  // namespace ir
