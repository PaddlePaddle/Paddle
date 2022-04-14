// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace infrt {
namespace dialect {

::mlir::Value Conv2dBnFuse_CreateFilter(
    ::mlir::PatternRewriter& rewriter,  // NOLINT
    ::mlir::Location& loc,              // NOLINT
    ::mlir::Value filter_attr,
    ::mlir::Value variance_attr,
    ::mlir::Value scale_attr,
    ::mlir::FloatAttr epsilon_attr);

::mlir::Value Conv2dBnFuse_CreateBias(
    ::mlir::PatternRewriter& rewriter,  // NOLINT
    ::mlir::Location& loc,              // NOLINT
    ::mlir::Value mean_attr,
    ::mlir::Value variance_attr,
    ::mlir::Value scale_attr,
    ::mlir::Value bias_attr,
    ::mlir::FloatAttr epsilon_attr);

}  // namespace dialect
}  // namespace infrt
