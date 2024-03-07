#pragma once

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

// ElementWise/Broadcast ops which have shardable dimentions and reduction ancestors.
template <typename T>
struct PartialShardablePattern {};

// Reduce base pattern
template <typename T>
struct ReductionPattern {
  using Nothing = std::monostate;
  std::variant<Nothing, InjectiveSourcePattern<T>, PartialShardablePattern<T>> opt_inputs;
  SingleReductionOpPattern<T> reduction_op_pattern;
};

// Stmt := IS | R | PS
// ops in StmtPattern will be lowered into a inlined cuda code.
template <typename T>
using StmtPattern = std::variant<InjectiveSourcePattern<T>, ReductionPattern<T>, PartialShardablePattern<T>>;

// Stmts := [Stmt]
template <typename T>
using StmtsPattern = std::vector<StmtPattern>;

// fuse rules:
//  1. PS * PS -> PS
//  2. IS * PS -> PS
//  3. IS * R -> R
//  4. PS * R -> R

// lifting rules:
//  1. R -> Stmts
//  2. PS -> Stmts
//  3. Stmts * Stmts -> Stmts

// OpTopoPattern := Error | Stmts
template <typename T>
using OpTopoPattern = std::variant<ErrorPattern<T>, StmtsPattern<T>>;

}
