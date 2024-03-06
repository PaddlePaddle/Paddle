#pragma once

#include <vector>

namespace cinn::api {

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


// SR := [R | PS]
template <typename T>
using ShardableReductionsPattern = std::vector<std::variant<ReductionPattern<T>, PartialShardablePattern<T>>>;

// fuse rules:
//  1. PS * PS -> PS
//  2. IS * PS -> PS
//  3. IS * R -> R
//  4. PS * R -> R

// lifting rules:
//  1. R -> SR
//  2. PS -> SR
//  3. SR * SR -> SR

// OpTopoPattern := IS | SR
template <typename T>
using OpTopoPattern = std::variant<InjectiveSourcePattern<T>, ShardableReductionsPattern<T>>;

}
