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

/// The design and code is mainly from MLIR, thanks to the great project.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/op_info.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace pir {

class IR_API FrozenRewritePatternSet {
  using NativePatternListT = std::vector<std::unique_ptr<RewritePattern>>;

 public:
  using OpSpecificNativePatternListT =
      std::unordered_map<OpInfo, std::vector<RewritePattern*>>;

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
      const std::vector<std::string>& disabled_pattern_labels = {},
      const std::vector<std::string>& enabled_pattern_labels = {});

  /// Return the op specific native patterns held by this list.
  const OpSpecificNativePatternListT& op_specific_native_patterns() const {
    return impl_->op_specific_native_pattern_map_;
  }

  /// Return the "match any" native patterns held by this list.
  const NativePatternListT& match_any_op_native_patterns() const {
    return impl_->match_any_op_native_patterns_;
  }

 private:
  struct Impl {
    OpSpecificNativePatternListT op_specific_native_pattern_map_;

    NativePatternListT op_specific_native_patterns_;

    NativePatternListT match_any_op_native_patterns_;
  };

  std::shared_ptr<Impl> impl_;
};

}  // namespace pir
