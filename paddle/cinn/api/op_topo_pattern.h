#pragma once

#include <vector>

namespace cinn::api {

// ElementWise/Broadcast/Injective Ops without reduction ancestors.
template <typename T>
struct InjectiveSourcePattern {};

// Reduce ops
template <typename T>
struct ReductionPattern {};

// ElementWise/Broadcast ops which have shardable dimentions and reduction ancestors.
template <typename T>
struct PartialShardablePattern {};

template <typename T>
using ShardableReductionsPattern = std::vector<std::variant<ReductionPattern<T>, PartialShardablePattern<T>>>;

// fuse rules:
//  1. IS * PS -> PS
//  2. PS * PS -> PS
//  3. PS * R -> R
//  4. IS * R -> R

// lifting rules:
//  1. R -> SR
//  2. PS -> SR
//  3. SR * SR -> SR

// OpTopoPattern := IS | SR
template <typename T>
using OpTopoPattern = std::variant<InjectiveSourcePattern<T>, ShardableReductionsPattern<T>>;

}
