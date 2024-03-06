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

// SR := [R | PS]
template <typename T>
using ShardableReductionsPattern = std::vector<std::variant<ReductionPattern<T>, PartialShardablePattern<T>>>;

// Compose rules:
//  1. IS * PS -> PS
//  2. PS * PS -> PS
//  3. R * PS -> RS
//  4. RS * (PS | R) -> RS

// OpsTopoPattern := IS | SR
template <typename T>
using OpsTopoPattern = std::variant<InjectiveSourcePattern<T>, ShardableReductionsPattern<T>>;

}
