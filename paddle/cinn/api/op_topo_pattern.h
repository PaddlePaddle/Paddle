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

#include <list>
#include <variant>
#include <vector>

namespace cinn::api {

template <typename T>
struct ErrorPattern {};

// ElementWise/Broadcast/Injective Ops without reduction ancestors.
template <typename T>
struct InjectiveSourcePattern {};

// Reduce op
template <typename T>
struct SingleReductionOpPattern {};

// ElementWise/Broadcast ops which have shardable dimentions and reduction
// ancestors.
template <typename T>
struct PartialShardablePattern {};

// Reduce base pattern
template <typename T>
struct ReductionPattern {
  using Nothing = std::monostate;
  std::variant<Nothing, InjectiveSourcePattern<T>, PartialShardablePattern<T>>
      input;
  SingleReductionOpPattern<T> reduce_op_pattern;

  bool HasFusedInput() const {
    return !std::holds_alternative<Nothing>(this->input);
  }
};

// Stmt := IS | R | PS
// ops in StmtPattern will be lowered into a inlined cuda code.
template <typename T>
using StmtPattern = std::variant<InjectiveSourcePattern<T>,
                                 ReductionPattern<T>,
                                 PartialShardablePattern<T>>;

// Stmts := [Stmt]
template <typename T>
using StmtPatternVec = std::vector<StmtPattern<T>>;
// fuse rules:
//  1. IS * IS -> IS
//  2. PS * PS -> PS
//  3. IS * PS -> PS
//  4. IS * R -> R
//  5. PS * R -> R
// lifting rules:
//  1. R -> Stmts
//  2. PS -> Stmts
//  3. Stmts * Stmts -> Stmts
// OpTopoPattern := Error | Stmts

template <typename T>
using OpTopoPattern = std::variant<ErrorPattern<T>, StmtPatternVec<T>>;

}  // namespace cinn::api
