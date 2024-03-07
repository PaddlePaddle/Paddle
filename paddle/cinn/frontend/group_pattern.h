#pragma once

#include <unordered_map>
#include "paddle/cinn/api/op_topo_pattern.h"
#include "paddle/pir/include/core/operation.h"

namespace cinn::frontend {

struct FrontendPattern {};

}

namespace cinn::api {

template<>
struct ErrorPattern<frontend::FrontendPattern> {
  const pir::Operation* op;
  std::string error_string;
};

template<>
struct InjectiveSourcePattern<frontend::FrontendPattern> {
  std::vector<const pir::Operation*> ops;
};

template<>
struct SingleReductionOpPattern<frontend::FrontendPattern> {
  const pir::Operation* reduce_op;
};

struct ShardableAxes {
  int axis;
  std::string axis_name;
};

struct ShardableAxesSignature {
  using OpOperand = std::pair<const pir::Operation*, /*operand index*/int>;

  std::vector<ShardableAxes> output_shardable_axes;
  std::unordered_map<OpOperand, ShardableAxes> input_shardable_axes;
};

template<>
struct PartialShardablePattern<frontend::FrontendPattern> {
  std::vector<const pir::Operation*> ops;
  ShardableAxesSignature shardable_axes_signature;
};

}

namespace cinn::frontend {

using GroupPattern = api::OpTopoPattern<FrontendPattern>;
using ErrorGroupPattern = api::ErrorPattern<FrontendPattern>;

}