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

/// The design and code is mainly from MLIR, thanks to the greate project.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/IR/OperationSupport.h"

namespace infra {

class FrozenRewritePatternSet {
  using NativePatternListT = std::vector<std::unique_ptr<RewritePattern>>;

 public:
  using OpSpecificNativePatternListT =
      llvm::DenseMap<mlir::OperationName, std::vector<RewritePattern*>>;

  FrozenRewritePatternSet();
  FrozenRewritePatternSet(FrozenRewritePatternSet&& patterns) = default;
  FrozenRewritePatternSet(const FrozenRewritePatternSet& patterns) = default;
  FrozenRewritePatternSet& operator=(FrozenRewritePatternSet&& patterns) =
      default;
  FrozenRewritePatternSet& operator=(const FrozenRewritePatternSet& patterns) =
      default;
  ~FrozenRewritePatternSet() = default;

  /// Freeze the patterns held in `patterns`, and take ownership.
  FrozenRewritePatternSet(
      RewritePatternSet&& patterns,
      llvm::ArrayRef<std::string> disabled_pattern_labels = std::nullopt,
      llvm::ArrayRef<std::string> enabled_pattern_labels = std::nullopt);

  /// Return the op specific native patterns held by this list.
  const OpSpecificNativePatternListT& GetOpSpecificNativePatterns() const {
    return impl_->native_op_specific_pattern_map;
  }

  /// Return the "match any" native patterns held by this list.
  llvm::iterator_range<
      llvm::pointee_iterator<NativePatternListT::const_iterator>>
  GetMatchAnyOpNativePatterns() const {
    const NativePatternListT& native_list = impl_->native_any_op_patterns;
    return llvm::make_pointee_range(native_list);
  }

 private:
  struct Impl {
    OpSpecificNativePatternListT native_op_specific_pattern_map;

    NativePatternListT native_op_specific_pattern_list;

    NativePatternListT native_any_op_patterns;
  };

  std::shared_ptr<Impl> impl_;
};

}  // namespace infra
