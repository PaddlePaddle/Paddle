#pragma once

#include <unordered_map>
#include <atomic>
#include <vector>
#include "paddle/cinn/api/op_topo_pattern.h"
#include "paddle/pir/include/core/operation.h"

namespace cinn::frontend {

struct FrontendPattern {};

}

namespace cinn::api {

template<>
struct ErrorPattern<frontend::FrontendPattern> {
  explicit ErrorPattern(const ErrorPattern<frontend::FrontendPatterns>& other) = default;

  std::vector<const pir::Operation*> ops;
  std::string error_string;
};

template<>
struct InjectiveSourcePattern<frontend::FrontendPattern> {
  explicit InjectiveSourcePattern(const InjectiveSourcePattern<frontend::FrontendPatterns>& other) = default;
  std::vector<const pir::Operation*> ops;
};

template<>
struct SingleReductionOpPattern<frontend::FrontendPattern> {
  explicit SingleReductionOpPattern(const SingleReductionOpPattern<frontend::FrontendPatterns>& other) = default;
  const pir::Operation* reduce_op;
};

struct ShardableAxis {
  int axis;
  std::optional<std::string> axis_name;

  bool operator==(const ShardableAxis& other) const {
    return this->axis == other.axis && this->axis_name == other.axis_name;
  }

  static int64_t UnqiueSeqNo() {
    static std::atomic<int64_t> cnt(0);
    return ++cnt;
  }
};

using ShardableAxes = std::vector<ShardableAxis>;

struct ShardableAxesUtil {
  using OldName2NewName = std::unorderd_map<std::string, std::string>;

  static OldName2NewName GetOldName2NewName(const ShardableAxes& old_sa, const ShardableAxes& new_sa) {
    OldName2NewName old_name2new_name;
    for (const auto& [old_axis, old_name] : old_sa) {
      for (const auto& [new_axis, new_name] : new_sa) {
        if (old_axis == new_axis) {
          CHECK(old_name2new_name.emplace(old_name, new_name).second);
        }
      }
    }
    return old_name2new_name;
  }

  static void UpdateShardableAxes(const OldName2NewName& old2new, ShardableAxes* sa) {
    for (auto iter = sa->begin(); iter != sa->end();) {
      const auto& pair_it = old2new.find(iter->axis_name);
      if (pair_it != old2new.end()) {
        iter->axis_name = pair_it.second;
        ++iter; 
      } else {
        iter = sa->erase(iter); 
      }
    }
  }

  static ShardableAxes GetCommonShardableAxes(const ShardableAxes& lhs, const ShardableAxes& rhs) {
    ShardableAxes ret;
    for (const auto& lhs_axis : lhs) {
      for (const auto& rhs_axis : rhs) {
        if (lhs_axis == rhs_axis) {
          ret.emplace_back(lhs_axis);
        }
      }
    }
    return ret;
  }

  static ShardableAxes GetFullyShardableAxes(size_t rank) {
    ShardableAxes ret;
    for (int i = 0; i < rank; ++i) {
      ret.emplace_back(ShardableAxis{
        .axis=i,
        .axis_name=std::string("D") + std::to_string(ShardableAxis::UnqiueSeqNo()),
      });
    }
    return ret;
  }
};

struct ShardableAxesSignature {
  using OpOperand = std::pair<const pir::Operation*, /*operand index*/int>;

  ShardableAxes output_shardable_axes;
  std::unordered_map<OpOperand, ShardableAxes> input_shardable_axes;
};

template<>
struct PartialShardablePattern<frontend::FrontendPattern> {
  explicit PartialShardablePattern(const PartialShardablePattern<frontend::FrontendPatterns>& other) = default;

  std::vector<const pir::Operation*> ops;
  ShardableAxesSignature shardable_axes_signature;
};

}

namespace cinn::frontend {

using GroupPattern = api::OpTopoPattern<FrontendPattern>;
using ErrorGroupPattern = api::ErrorPattern<FrontendPattern>;

}