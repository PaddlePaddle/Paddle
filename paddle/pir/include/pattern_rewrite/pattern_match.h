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

// The design is mainly from MLIR, very thanks to the great project.

#pragma once

#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_info.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/type_id.h"
#include "paddle/pir/include/core/type_name.h"
#include "paddle/pir/include/core/value.h"

namespace pir {

// This class represents the benefit of a pattern. The most common
// unit to use is the `number of operations` in the pattern.
class IR_API PatternBenefit {
 public:
  PatternBenefit() = default;
  PatternBenefit(uint32_t val) : val_(val) {}  // NOLINT

  uint32_t benefit() { return val_; }

  bool operator==(const PatternBenefit& rhs) const { return val_ == rhs.val_; }
  bool operator!=(const PatternBenefit& rhs) const { return !(*this == rhs); }
  bool operator<(const PatternBenefit& rhs) const { return val_ < rhs.val_; }
  bool operator>(const PatternBenefit& rhs) const { return rhs < *this; }
  bool operator<=(const PatternBenefit& rhs) const { return !(*this > rhs); }
  bool operator>=(const PatternBenefit& rhs) const { return !(*this < rhs); }

 private:
  uint32_t val_{0};
};

// This class contains all of the data related to a Pattern, but not contains
// any methods for the matching. This class is used to interface with the
// metadata of a pattern, such as benefit or root operation.
class IR_API Pattern {
  enum class RootKind {
    // The pattern root matches "any" operation.
    Any,
    // The pattern root is matched using a concrete operation.
    OperationInfo,
    // The pattern root is matched using an interface id.
    InterfaceId,
    // The patter root is matched using a trait id.
    TraitId
  };

 public:
  const std::vector<OpInfo>& generated_ops() const { return generated_ops_; }

  std::optional<OpInfo> root_kind() const {
    if (root_kind_ == RootKind::OperationInfo)
      return OpInfo::RecoverFromVoidPointer(root_val_);
    return std::nullopt;
  }

  std::optional<TypeId> GetRootInterfaceID() const {
    if (root_kind_ == RootKind::InterfaceId)
      return TypeId::RecoverFromVoidPointer(root_val_);
    return std::nullopt;
  }

  std::optional<TypeId> GetRootTraitID() const {
    if (root_kind_ == RootKind::TraitId)
      return TypeId::RecoverFromVoidPointer(root_val_);
    return std::nullopt;
  }

  PatternBenefit benefit() const { return benefit_; }

  IrContext* ir_context() const { return context_; }

  std::string debug_name() const { return debug_name_; }

  void SetDebugName(const std::string& name) { debug_name_ = name; }

  const std::vector<std::string>& debug_labels() const { return debug_labels_; }

  void AddDebugLabels(const std::vector<std::string>& labels) {
    debug_labels_.insert(debug_labels_.end(), labels.begin(), labels.end());
  }

  void AddDebugLabels(const std::string& label) {
    debug_labels_.push_back(label);
  }

 protected:
  struct MatchAnyOpTypeTag {};
  struct MatchInterfaceOpTypeTag {};
  struct MatchTraitOpTypeTag {};

  Pattern(const std::string& root_name,
          PatternBenefit benefit,
          IrContext* context,
          const std::vector<std::string>& generated_names = {});

  Pattern(MatchAnyOpTypeTag tag,
          PatternBenefit benefit,
          IrContext* context,
          const std::vector<std::string>& generated_names = {});

  Pattern(MatchInterfaceOpTypeTag tag,
          TypeId interface_id,
          PatternBenefit benefit,
          IrContext* context,
          const std::vector<std::string>& generated_names = {});

  Pattern(MatchTraitOpTypeTag tag,
          TypeId trait_id,
          PatternBenefit benefit,
          IrContext* context,
          const std::vector<std::string>& generated_names = {});

 private:
  Pattern(void* root_val,
          RootKind root_kind,
          const std::vector<std::string>& generated_names,
          PatternBenefit benefit,
          IrContext* context);

  void* root_val_;
  RootKind root_kind_;

  const PatternBenefit benefit_;
  IrContext* context_;
  // A list of the potential operations that may be generated when rewriting an
  // op with this pattern.
  std::vector<OpInfo> generated_ops_;

  std::string debug_name_;
  std::vector<std::string> debug_labels_;
};

class PatternRewriter;

class IR_API RewritePattern : public Pattern {
 public:
  virtual ~RewritePattern();

  virtual void Rewrite(Operation* op,
                       PatternRewriter& rewriter) const {  // NOLINT
    IR_THROW(
        "need to implement either MatchAndRewrite or one of the rewrite "
        "functions.");
  }

  virtual bool Match(Operation* op) const {
    IR_THROW("need to implement either MatchAndRewrite or Match.");
    return false;
  }

  virtual bool MatchAndRewrite(Operation* op,
                               PatternRewriter& rewriter) const {  // NOLINT
    if (Match(op)) {
      Rewrite(op, rewriter);
      return true;
    }
    return false;
  }

  virtual void Initialize() {}

  template <typename T, typename... Args>
  static std::unique_ptr<T> Create(Args&&... args) {
    std::unique_ptr<T> pattern =
        std::make_unique<T>(std::forward<Args>(args)...);
    pattern->Initialize();

    if (pattern->debug_name().empty())
      pattern->SetDebugName(pir::get_type_name<T>());
    return pattern;
  }

 protected:
  using Pattern::Pattern;
};

namespace detail {
// A wrapper around PatternWrite that allows for matching and rewriting
// against an instance of a derived operation class or Interface.
template <typename SourceOp>
struct OpOrInterfaceRewritePatternBase : public RewritePattern {
  using RewritePattern::RewritePattern;

  void Rewrite(Operation* op,
               PatternRewriter& rewriter) const final {  // NOLINT
    Rewrite(op->dyn_cast<SourceOp>(), rewriter);
  }

  bool Match(Operation* op) const final {
    return Match(op->dyn_cast<SourceOp>());
  }
  bool MatchAndRewrite(Operation* op,
                       PatternRewriter& rewriter) const final {  // NOLINT
    return MatchAndRewrite(op->dyn_cast<SourceOp>(), rewriter);
  }

  virtual void Rewrite(SourceOp op,
                       PatternRewriter& rewriter) const {  // NOLINT
    IR_THROW("must override Rewrite or MatchAndRewrite");
  }
  virtual bool Match(SourceOp op) const {
    IR_THROW("must override Match or MatchAndRewrite");
  }
  virtual bool MatchAndRewrite(SourceOp op,
                               PatternRewriter& rewriter) const {  // NOLINT
    if (Match(op)) {
      Rewrite(op, rewriter);
      return true;
    }
    return false;
  }
};
}  // namespace detail

// OpRewritePattern is a wrapper around RewritePattern that allows for
// matching and rewriting against an instance of a derived operation
// class as opposed to a raw Operation.
template <typename SourceOp>
struct OpRewritePattern
    : public detail::OpOrInterfaceRewritePatternBase<SourceOp> {
  OpRewritePattern(IrContext* context,
                   PatternBenefit benefit = 1,
                   const std::vector<std::string>& generated_names = {})
      : detail::OpOrInterfaceRewritePatternBase<SourceOp>(
            SourceOp::name(), benefit, context, generated_names) {}
};

// TODO(wilber): Support OpInterfaceRewritePattern and OpTraitRewritePattern.
// ...

// This class provides a series of interfaces for modifying IR and tracking IR
// changes. This class provides a unified API for IR modification.
class RewriterBase : public Builder {
 public:
  // TODO(wilber): Supplementary methods of block and region.

  virtual void ReplaceOpWithIf(Operation* op,
                               const std::vector<Value>& new_values,
                               bool* all_uses_replaced,
                               const std::function<bool(OpOperand)>& functor);

  void ReplaceOpWithIf(Operation* op,
                       const std::vector<Value>& new_values,
                       const std::function<bool(OpOperand)>& functor);

  virtual void ReplaceOp(Operation* op, const std::vector<Value>& new_values);

  // Replaces the result op with a new op.
  // The result values of the two ops must be the same types.
  template <typename OpTy, typename... Args>
  OpTy ReplaceOpWithNewOp(Operation* op, Args&&... args) {
    auto new_op = Build<OpTy>(std::forward<Args>(args)...);
    ReplaceOpWithResultsOfAnotherOp(op, new_op.operation());
    return new_op;
  }

  // This method erases an operation that is known to have no uses.
  virtual void EraseOp(Operation* op);

  IR_API void ReplaceAllUsesWith(Value from, Value to);

  void ReplaceUseIf(Value from,
                    Value to,
                    std::function<bool(OpOperand&)> functor);

 protected:
  explicit RewriterBase(IrContext* ctx) : Builder(ctx) {}

  virtual ~RewriterBase();

  virtual void NotifyRootReplaced(Operation* op,
                                  const std::vector<Value>& replacement) {}

  virtual void NotifyOperationRemoved(Operation* op) {}

  virtual void NotifyOperationInserted(Operation* op) {}

  virtual void StartRootUpdate(Operation* op) {}

  virtual void FinalizeRootUpdate(Operation* op) {}

  virtual void CancelRootUpdate(Operation* op) {}

  template <typename CallableT>
  void UpdateRootInplace(Operation* root, CallableT&& callable) {
    StartRootUpdate(root);
    callable();
    FinalizeRootUpdate(root);
  }

 private:
  void operator=(const RewriterBase&) = delete;
  RewriterBase(const RewriterBase&) = delete;

  void ReplaceOpWithResultsOfAnotherOp(Operation* op, Operation* new_op);
};

class PatternRewriter : public RewriterBase {
 public:
  using RewriterBase::RewriterBase;
};

// A pattern collection, easy to add patterns.
class RewritePatternSet {
  using NativePatternListT = std::vector<std::unique_ptr<RewritePattern>>;

 public:
  explicit RewritePatternSet(IrContext* context) : context_(context) {}

  // Construct a RewritePatternSet with the given patterns.
  RewritePatternSet(IrContext* context, std::unique_ptr<RewritePattern> pattern)
      : context_(context) {
    native_patterns_.emplace_back(std::move(pattern));
  }

  IrContext* ir_context() const { return context_; }

  NativePatternListT& native_patterns() { return native_patterns_; }

  void Clear() { native_patterns_.clear(); }

  bool Empty() const { return native_patterns_.empty(); }

  // 'add' methods for adding patterns to the set.
  template <typename... Ts,
            typename ConstructorArg,
            typename... ConstructorArgs,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  RewritePatternSet& Add(ConstructorArg&& arg, ConstructorArgs&&... args) {
    (void)std::initializer_list<int>{
        (AddImpl<Ts>({},
                     std::forward<ConstructorArg>(arg),
                     std::forward<ConstructorArgs>(args)...),
         0)...};
    return *this;
  }

  template <typename... Ts,
            typename ConstructorArg,
            typename... ConstructorArgs,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  RewritePatternSet& AddWithLabel(const std::vector<std::string>& debug_labels,
                                  ConstructorArg&& arg,
                                  ConstructorArgs&&... args) {
    (void)std::initializer_list<int>{
        (AddImpl<Ts>(debug_labels,
                     std::forward<ConstructorArg>(arg),
                     std::forward<ConstructorArgs>(args)...),
         0)...};
    return *this;
  }

  RewritePatternSet& Add(std::unique_ptr<RewritePattern> pattern) {
    native_patterns_.emplace_back(std::move(pattern));
    return *this;
  }

 private:
  template <typename T, typename... Args>
  std::enable_if_t<std::is_base_of<RewritePattern, T>::value> AddImpl(
      const std::vector<std::string>& debug_labels, Args&&... args) {
    std::unique_ptr<T> pattern =
        RewritePattern::Create<T>(std::forward<Args>(args)...);
    pattern->AddDebugLabels(debug_labels);
    native_patterns_.emplace_back(std::move(pattern));
  }

 private:
  IrContext* const context_;

  NativePatternListT native_patterns_;
};

}  // namespace pir
