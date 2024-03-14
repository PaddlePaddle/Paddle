#pragma once

#include <unordered_map>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <variant>
#include "paddle/cinn/api/op_topo_pattern.h"
#include "paddle/pir/include/core/operation.h"
#include "glog/logging.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/tree.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace cinn::frontend {

struct OpAndOperandIndex {
  const pir::Operation* op;
  const int operand_index;

  bool operator==(const OpAndOperandIndex& other) const {
    return this->op == other.op && this->operand_index == other.operand_index;
  }
};

}

namespace std {

template<>
struct hash<cinn::frontend::OpAndOperandIndex> {

  size_t operator()(const cinn::frontend::OpAndOperandIndex& op_operand) const {
    return cinn::adt::hash_combine(std::hash<const pir::Operation*>()(op_operand.op), op_operand.operand_index);
  }
};

}

namespace cinn::frontend {

struct FrontendPattern {};

struct ShardableAxis {
  int axis;
  std::string axis_name;

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
  using OldName2NewName = std::unordered_map<std::string, std::string>;

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
        iter->axis_name = pair_it->second;
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
  ShardableAxes output_shardable_axes;
  std::unordered_map<OpAndOperandIndex, ShardableAxes> input_shardable_axes;
};

}

namespace cinn::api {

template<>
struct ErrorPattern<frontend::FrontendPattern> {
  std::vector<const pir::Operation*> ops;
  std::string error_string;
};

template<>
struct InjectiveSourcePattern<frontend::FrontendPattern> {
  std::vector<const pir::Operation*> ops;
  const pir::Operation* sole_sink;
};

template<>
struct SingleReductionOpPattern<frontend::FrontendPattern> {  
  const pir::Operation* reduce_op;
};
template<>
struct PartialShardablePattern<frontend::FrontendPattern> {
  std::vector<const pir::Operation*> ops;
  const pir::Operation* sole_sink;
  frontend::ShardableAxesSignature shardable_axes_signature;
};

}

namespace cinn::frontend {

using ErrorGroupPattern = api::ErrorPattern<frontend::FrontendPattern>;
using GroupPattern = api::OpTopoPattern<frontend::FrontendPattern>;

struct LoopAlignableStmtsPattern {
  std::vector<api::StmtPattern<frontend::FrontendPattern>> stmts;
};

struct ClusteringResult {
  std::vector<LoopAlignableStmtsPattern> loop_alignable_list;
};

}