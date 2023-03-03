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

// The design and code is mainly from MLIR, very thanks to the greate project.

#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TypeName.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
// #include "utils/STLExtras.h"

namespace infra {

class PatternRewriter;

class PatternBenefit {
  enum { ImpossibleToMatchSentinel = std::numeric_limits<unsigned>::max() };

 public:
  PatternBenefit() = default;
  PatternBenefit(unsigned benefit);  // NOLINT

  bool IsImpossibleToMatch() const { return *this == PatternBenefit(); }
  unsigned GetBenefit() { return representation; }

  bool operator==(const PatternBenefit& rhs) const {
    return representation == rhs.representation;
  }
  bool operator!=(const PatternBenefit& rhs) const { return !(*this == rhs); }
  bool operator<(const PatternBenefit& rhs) const {
    return representation < rhs.representation;
  }
  bool operator>(const PatternBenefit& rhs) const { return rhs < *this; }
  bool operator<=(const PatternBenefit& rhs) const { return !(*this > rhs); }
  bool operator>=(const PatternBenefit& rhs) const { return !(*this < rhs); }

 private:
  unsigned int representation{ImpossibleToMatchSentinel};
};

//==----==//
// Pattern
//==----==//
class Pattern {
  enum class RootKind {
    Any,
    OperationName,
    InterfaceID,
    TraitID,
  };

 public:
  llvm::ArrayRef<mlir::OperationName> GetGeneratedOps() const {
    return generated_ops_;
  }

  std::optional<mlir::OperationName> GetRootKind() const {
    if (root_kind_ == RootKind::OperationName)
      return mlir::OperationName::getFromOpaquePointer(root_val_);
    return std::nullopt;
  }

  std::optional<mlir::TypeID> GetRootInterfaceID() const {
    if (root_kind_ == RootKind::InterfaceID)
      return mlir::TypeID::getFromOpaquePointer(root_val_);
    return std::nullopt;
  }

  std::optional<mlir::TypeID> GetRootTraitID() const {
    if (root_kind_ == RootKind::TraitID)
      return mlir::TypeID::getFromOpaquePointer(root_val_);
    return std::nullopt;
  }

  PatternBenefit GetBenefit() const { return benefit; }

  /// Returns true if this pattern is known to result in recursive application,
  /// i.e. this pattern may generate IR that also matches this pattern, but is
  /// known to bound the recursion. This is signals to a rewrite driver that it
  /// is safe to apply this pattern recursively to generate IR.
  bool HasBoundRewriteRecursion() const { return has_bound_recursion_; }

  mlir::MLIRContext* GetContext() const { return context_; }

  llvm::StringRef GetDebugName() const { return debug_name_; }

  void SetDebugName(llvm::StringRef name) { debug_name_ = name; }

  llvm::ArrayRef<llvm::StringRef> GetDebugLabels() const {
    return debug_labels_;
  }

  void AddDebugLabels(llvm::ArrayRef<llvm::StringRef> labels) {
    debug_labels_.insert(debug_labels_.end(), labels.begin(), labels.end());
  }

  void AddDebugLabels(llvm::StringRef label) { debug_labels_.push_back(label); }

 protected:
  struct MatchAnyOpTypeTag {};
  struct MatchInterfaceOpTypeTag {};
  struct MatchTraitOpTypeTag {};

  Pattern(llvm::StringRef root_name,
          PatternBenefit benefit,
          mlir::MLIRContext* context,
          llvm::ArrayRef<llvm::StringRef> generated_names = {});
  Pattern(MatchAnyOpTypeTag tag,
          PatternBenefit benefit,
          mlir::MLIRContext* context,
          llvm::ArrayRef<llvm::StringRef> generated_names = {});
  Pattern(MatchInterfaceOpTypeTag tag,
          mlir::TypeID interface_id,
          PatternBenefit benefit,
          mlir::MLIRContext* context,
          llvm::ArrayRef<llvm::StringRef> generated_names = {});
  Pattern(MatchTraitOpTypeTag tag,
          mlir::TypeID trait_id,
          PatternBenefit benefit,
          mlir::MLIRContext* context,
          llvm::ArrayRef<llvm::StringRef> generated_names = {});

  void SetHasBoundedRewriteRecursion(bool has_bound_recursion = true) {
    has_bound_recursion_ = has_bound_recursion;
  }

 private:
  Pattern(const void* root_val,
          RootKind root_kind,
          llvm::ArrayRef<llvm::StringRef> generated_names,
          PatternBenefit benefit,
          mlir::MLIRContext* context);

  /// The value used to match the root operation of the pattern.
  const void* root_val_;
  RootKind root_kind_;

  /// The expected benefit of matching this pattern.
  const PatternBenefit benefit;
  mlir::MLIRContext* context_;
  bool has_bound_recursion_;
  std::vector<mlir::OperationName> generated_ops_;

  llvm::StringRef debug_name_;
  std::vector<llvm::StringRef> debug_labels_;
};

//==----==//
// RewritePattern
//==----==//

/// RewritePattern is the common base class for all DAG to DAG replacements.

class RewritePattern : public Pattern {
 public:
  virtual ~RewritePattern() = default;

  virtual void Rewrite(mlir::Operation* op,
                       PatternRewriter& rewriter) const {  // NOLINT
    llvm_unreachable(
        "need to implement either MatchAndRewrite or one of the rewrite "
        "functions.");
  }

  virtual mlir::LogicalResult Match(mlir::Operation* op) const {
    llvm_unreachable("need to implement either MatchAndRewrite or Match.");
  }

  virtual mlir::LogicalResult MatchAndRewrite(
      mlir::Operation* op,
      PatternRewriter& rewriter) const {  // NOLINT
    if (mlir::succeeded(Match(op))) {
      Rewrite(op, rewriter);
      return mlir::success();
    }
    return mlir::failure();
  }

  virtual void Initialize() {}

  ///
  template <typename T, typename... Args>
  static std::unique_ptr<T> Create(Args&&... args) {
    std::unique_ptr<T> pattern =
        std::make_unique<T>(std::forward<Args>(args)...);

    // InitializePattern<T>(*pattern);

    pattern->Initialize();

    if (pattern->GetDebugName().empty())
      pattern->SetDebugName(llvm::getTypeName<T>());
    return pattern;
  }

 protected:
  using Pattern::Pattern;

 private:
  // template <typename T, typename... Args>
  // using has_initialize = decltype(std::declval<T>().Initialize());
  // template <typename T>
  // using detect_has_initialize = infra::is_detected<has_initialize, T>;

  // template <typename T>
  // static std::enable_if_t<detect_has_initialize<T>::value> InitializePattern(
  //     T& pattern) {
  //   pattern.Initialize();
  // }
  // template <typename T>
  // static std::enable_if_t<!detect_has_initialize<T>::value>
  // InitializePattern(
  //     T& pattern) {}
};

namespace detail {
/// OpOrInterfaceRewritePatternBase is a wrapper around RewritePattern that
/// allows for matching and rewriting against an instance of a derived operation
/// class or Interface
template <typename SourceOp>
struct OpOrInterfaceRewritePatternBase : public RewritePattern {
  using RewritePattern::RewritePattern;

  void Rewrite(mlir::Operation* op,
               PatternRewriter& rewriter) const final {  // NOLINT
    Rewrite(llvm::cast<SourceOp>(op), rewriter);
  }
  mlir::LogicalResult Match(mlir::Operation* op) const final {
    return Match(llvm::cast<SourceOp>(op));
  }
  mlir::LogicalResult MatchAndRewrite(
      mlir::Operation* op,
      PatternRewriter& rewriter) const final {  // NOLINT
    return MatchAndRewrite(llvm::cast<SourceOp>(op), rewriter);
  }

  virtual void Rewrite(SourceOp op,
                       PatternRewriter& rewriter) const {  // NOLINT
    llvm_unreachable("must override Rewrite or MatchAndRewrite");
  }
  virtual mlir::LogicalResult Match(
      SourceOp op,
      PatternRewriter& rewriter) const {  // NOLINT
    llvm_unreachable("must override Match or MatchAndRewrite");
  }
  virtual mlir::LogicalResult MatchAndRewrite(
      SourceOp op,
      PatternRewriter& rewriter) const {  // NOLINT
    if (mlir::succeeded(Match(op))) {
      Rewrite(op, rewriter);
      return mlir::success();
    }
    return mlir::failure();
  }
};
}  // namespace detail

/// OpRewritePattern is a wrapper around RewritePattern that allows for
/// matching and rewriting against an instance of a derived operation class as
/// opposed to a raw Operation.
template <typename SourceOp>
struct OpRewritePattern
    : public detail::OpOrInterfaceRewritePatternBase<SourceOp> {
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching and a list of generated
  /// ops.
  OpRewritePattern(mlir::MLIRContext* context,
                   PatternBenefit benefit = 1,
                   llvm::ArrayRef<llvm::StringRef> generated_names = {})
      : detail::OpOrInterfaceRewritePatternBase<SourceOp>(
            SourceOp::getOperationName(), benefit, context, generated_names) {}
};

/// OpInterfaceRewritePattern is a wrapper around RewritePattern that allows for
/// matching and rewriting against an instance of an operation interface instead
/// of a raw Operation.
template <typename SourceOp>
struct OpInterfaceRewritePattern
    : public detail::OpOrInterfaceRewritePatternBase<SourceOp> {
  OpInterfaceRewritePattern(mlir::MLIRContext* context,
                            PatternBenefit benefit = 1)
      : detail::OpOrInterfaceRewritePatternBase<SourceOp>(
            Pattern::MatchInterfaceOpTypeTag(),
            SourceOp::getInterfaceID(),
            benefit,
            context) {}
};

/// OpTraitRewritePattern
/// TODO(wilber): ...

//==----==//
// RewriterBase
//==----==//

class RewriterBase : public mlir::OpBuilder, public mlir::OpBuilder::Listener {
 public:
  // TODO(wilber): drop a lot of region, block method.

  virtual void ReplaceOpWithIf(mlir::Operation* op,
                               mlir::ValueRange new_values,
                               bool* all_uses_replaced,
                               std::function<bool(mlir::OpOperand&)> functor);
  void ReplaceOpWithIf(mlir::Operation* op,
                       mlir::ValueRange new_values,
                       std::function<bool(mlir::OpOperand&)> functor);

  virtual void ReplaceOp(mlir::Operation* op, mlir::ValueRange new_values);

  // virtual void ReplaceOpWithNewOp()

  virtual void EraseOp(mlir::Operation* op);

  virtual void StartRootUpdate(mlir::Operation* op) {}
  virtual void FinalizeRootUpdate(mlir::Operation* op) {}
  virtual void CancelRootUpdate(mlir::Operation* op) {}

  template <typename CallableT>
  void UpdateRootInPlace(mlir::Operation* root, CallableT&& callable) {
    StartRootUpdate(root);
    callable();
    FinalizeRootUpdate(root);
  }

  void ReplaceAllUsesWith(mlir::Value from, mlir::Value to);

  void ReplaceUseIf(mlir::Value from,
                    mlir::Value to,
                    std::function<bool(mlir::OpOperand&)> functor);

 protected:
  explicit RewriterBase(mlir::MLIRContext* ctx) : mlir::OpBuilder(ctx, this) {}

  virtual ~RewriterBase() = default;

  virtual void notifyRootReplaced(mlir::Operation* op,
                                  mlir::ValueRange replacement) {}

  virtual void notifyOperationRemoved(mlir::Operation* op) {}

  // virtual mlir::LogicalResult notifyMatchFailure(mlir::Location loc,
  // function_ref)

 private:
  void operator=(const RewriterBase&) = delete;
  RewriterBase(const RewriterBase&) = delete;

  void ReplaceOpWithResultsOfAnotherOp(mlir::Operation* op,
                                       mlir::Operation* new_op);
};

//==----==//
// PatternRewriter
//==----==//
class PatternRewriter : public RewriterBase {
 public:
  using RewriterBase::RewriterBase;

  // virtual bool CanRecoverFromRewriteFailure() const { return false; }
};

//==----==//
// RewritePatternSet
//==----==//

class RewritePatternSet {
  using NativePatternListT = std::vector<std::unique_ptr<RewritePattern>>;

 public:
  explicit RewritePatternSet(mlir::MLIRContext* context) : context_(context) {}

  RewritePatternSet(mlir::MLIRContext* context,
                    std::unique_ptr<RewritePattern> pattern)
      : context_(context) {
    native_patterns_.emplace_back(std::move(pattern));
  }

  mlir::MLIRContext* GetContext() const { return context_; }

  NativePatternListT& GetNativePatterns() { return native_patterns_; }

  void Clear() { native_patterns_.clear(); }

  // 'add' methods for adding patterns to the set.
  template <typename... Ts,
            typename ConstructorArg,
            typename... ConstructorArgs,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  RewritePatternSet& Add(ConstructorArg&& arg, ConstructorArgs&&... args) {
    (AddImpl<Ts>(std::nullopt,
                 std::forward<ConstructorArg>(arg),
                 std::forward<ConstructorArgs>(args)...),
     ...);
  }

  template <typename... Ts,
            typename ConstructorArg,
            typename... ConstructorArgs,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  RewritePatternSet& AddWithLabel(llvm::ArrayRef<llvm::StringRef> debug_labels,
                                  ConstructorArg&& arg,
                                  ConstructorArgs&&... args) {
    (AddImpl<Ts>(debug_labels,
                 std::forward<ConstructorArg>(arg),
                 std::forward<ConstructorArgs>(args)...),
     ...);
  }

  RewritePatternSet& Add(std::unique_ptr<RewritePattern> pattern) {
    native_patterns_.emplace_back(std::move(pattern));
    return *this;
  }

 private:
  template <typename T, typename... Args>
  std::enable_if_t<std::is_base_of_v<RewritePattern, T>> AddImpl(
      llvm::ArrayRef<llvm::StringRef> debug_labels, Args&&... args) {
    std::unique_ptr<T> pattern =
        RewritePattern::Create<T>(std::forward<Args>(args)...);
    pattern->AddDebugLabels(debug_labels);
    native_patterns_.emplace_back(std::move(pattern));
  }

 private:
  mlir::MLIRContext* const context_;
  NativePatternListT native_patterns_;
};

}  // namespace infra
