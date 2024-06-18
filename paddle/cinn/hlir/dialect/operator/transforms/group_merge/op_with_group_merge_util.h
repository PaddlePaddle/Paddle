// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <limits.h>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/framework/pir/group.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace cinn {
namespace dialect {
namespace ir {
// alias OpPatternKind and pir::Group
using OpPatternKind = hlir::framework::OpPatternKind;
using Group = hlir::framework::pir::Group;

template <typename T = int64_t>
std::vector<T> GetVectorAttr(const ::pir::Operation* op,
                             const std::string& name) {
  auto& attr_map = op->attributes();
  PADDLE_ENFORCE(
      attr_map.count(name),
      phi::errors::PreconditionNotMet(
          "attr [%s] MUST in attribute map for [%s] op", name, op->name()));
  auto& val = attr_map.at(name);

  PADDLE_ENFORCE(val.isa<::pir::ArrayAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "axis Type MUST ArrayAttribute for [%s] op", op->name()));
  auto array_list = val.dyn_cast<::pir::ArrayAttribute>().AsVector();
  std::vector<T> vec_res;
  if (array_list.size() > 0) {
    PADDLE_ENFORCE_EQ(array_list[0].isa<::pir::Int64Attribute>(),
                      true,
                      phi::errors::Unimplemented(
                          "the 0th elementwise MUST be ir::Int64Attribute"));
    for (size_t i = 0; i < array_list.size(); ++i) {
      vec_res.push_back(array_list[i].dyn_cast<::pir::Int64Attribute>().data());
    }
  }
  return vec_res;
}

phi::DDim GetFirstInputShape(const ::pir::Operation* op);

const phi::DDim& GetValueShape(const ::pir::Value& value);

bool WithoutLastDimInReduce(const std::vector<int64_t>& inshape,
                            const std::vector<int64_t>& axes);

int GetSharedSize(::pir::Operation* op);

inline bool always_fuse(
    ::pir::Operation* producer,
    const std::shared_ptr<Group>& consumer,
    ::pir::ShapeConstraintIRAnalysis* shape_analysis) {  // NOLINT
  return true;
}

inline bool no_fuse(::pir::Operation* producer,
                    const std::shared_ptr<Group>& consumer,
                    ::pir::ShapeConstraintIRAnalysis* shape_analysis) {
  return false;
}

inline bool is_same_shape(::pir::Operation* producer,
                          const std::shared_ptr<Group>& consumer,
                          ::pir::ShapeConstraintIRAnalysis* shape_analysis) {
  auto master_op = consumer->master_ops.begin();
  return shape_analysis->IsShapeEqual(producer->result(0),
                                      (*master_op)->result(0));
}

inline bool is_same_size(::pir::Operation* producer,
                         const std::shared_ptr<Group>& consumer,
                         ::pir::ShapeConstraintIRAnalysis* shape_analysis) {
  auto master_op = consumer->master_ops.begin();
  return shape_analysis->IsSameNumel(producer->result(0),
                                     (*master_op)->result(0));
}

inline bool without_last_dimension_in_reduce(
    ::pir::Operation* producer, const std::shared_ptr<Group>& consumer) {
  auto in_shape = ::common::vectorize<int64_t>(GetFirstInputShape(producer));
  auto reduce_axes = GetVectorAttr(producer, "axis");
  return WithoutLastDimInReduce(in_shape, reduce_axes);
}

inline bool reduce_fuse_reduce(
    ::pir::Operation* producer,
    const std::shared_ptr<Group>& consumer,
    ::pir::ShapeConstraintIRAnalysis* shape_analysis) {
  ::pir::Operation* reducer = NULL;
  for (auto* master : consumer->master_ops) {
    if (hlir::framework::pir::CompatibleInfo::OpKind(*master) ==
        OpPatternKind::kReduction) {
      reducer = master;
      break;
    }
  }
  // check reduce has same input shape and output shape
  auto producer_input_shape =
      ::common::vectorize<int64_t>(GetValueShape(producer->operand_source(0)));
  auto producer_output_shape =
      ::common::vectorize<int64_t>(GetValueShape(producer->result(0)));

  auto reducer_input_shape =
      ::common::vectorize<int64_t>(GetValueShape(reducer->operand_source(0)));
  auto reducer_output_shape =
      ::common::vectorize<int64_t>(GetValueShape(reducer->result(0)));

  auto producer_reduce_axes = GetVectorAttr(producer, "axis");
  auto reducer_reduce_axes = GetVectorAttr(reducer, "axis");

  for (auto& dim : producer_reduce_axes) {
    // if dim = -1, set as shape.size() - 1
    if (dim < 0) {
      dim += producer_input_shape.size();
    }
  }

  for (auto& dim : reducer_reduce_axes) {
    // if dim = -1,  set as shape.size() - 1
    if (dim < 0) {
      dim += reducer_input_shape.size();
    }
  }

  if (producer_output_shape == reducer_output_shape &&
      producer_reduce_axes == reducer_reduce_axes) {
    bool input_shape_same = producer_input_shape == reducer_input_shape;
    bool without_last_dim =
        WithoutLastDimInReduce(producer_input_shape, producer_reduce_axes) &&
        WithoutLastDimInReduce(reducer_input_shape, reducer_reduce_axes);
    // check shape is same
    if (input_shape_same || without_last_dim) {
      auto shared_size = GetSharedSize(producer);
      for (auto* master : consumer->master_ops) {
        if (hlir::framework::pir::CompatibleInfo::OpKind(*master) ==
            OpPatternKind::kReduction) {
          shared_size += GetSharedSize(master);
        }
      }

      constexpr int MAX_AVAILABLE_SHREAD = 32 * 1024;
      if (shared_size > MAX_AVAILABLE_SHREAD) {
        return false;
      }
      return true;
    }
  }

  return false;
}

inline bool is_horizontal_relation(::pir::Operation* producer,
                                   const std::shared_ptr<Group>& consumer) {
  auto check_dependency = [&](::pir::Operation* op) {
    std::queue<::pir::Operation*> candidates;
    std::unordered_set<::pir::Operation*> visited_set;
    candidates.push(op);

    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      // visit all producer op
      for (size_t i = 0; i < candidate->num_operands(); ++i) {
        auto tmp_op = candidate->operand_source(i).defining_op();
        // check dependency.
        if (producer == tmp_op) {
          return true;
        }
        // check op is in region.
        if (!consumer->ops_set.count(tmp_op)) {
          continue;
        }
        // recorded visited op.
        if (!visited_set.count(tmp_op)) {
          visited_set.insert(tmp_op);
          candidates.push(tmp_op);
        }
      }
    }

    return false;
  };

  for (auto op : consumer->ops_set) {
    if (hlir::framework::pir::CompatibleInfo::OpKind(*op) !=
        consumer->op_pattern_kind) {
      continue;
    }
    if (check_dependency(op)) {
      return false;
    }
  }

  return true;
}

inline bool horizontal_or_vertical_reduce_relation(
    ::pir::Operation* producer,
    const std::shared_ptr<Group>& consumer,
    ::pir::ShapeConstraintIRAnalysis* shape_analysis) {
  // check is same shape with horizontal relation.
  if (is_same_size(producer, consumer, shape_analysis)) {
    return true;
  }

  // reducer op in fusion op.
  ::pir::Operation* reducer = NULL;
  for (auto* master : consumer->master_ops) {
    if (hlir::framework::pir::CompatibleInfo::OpKind(*master) ==
        OpPatternKind::kReduction) {
      reducer = master;
      break;
    }
  }

  // check producer has same shape with reducer op.
  auto reduce_shape = ::common::vectorize(GetFirstInputShape(reducer));
  auto reduce_axes = GetVectorAttr(reducer, "axis");
  if (reduce_axes.empty()) {
    for (size_t i = 0; i < reduce_shape.size(); ++i) {
      reduce_axes.push_back(i);
    }
  }

  for (auto& axis : reduce_axes) {
    // if axis = -1, set as shape.size() - 1
    if (axis < 0) {
      axis += reduce_shape.size();
    }
  }

  auto op_shape =
      ::common::vectorize<int64_t>(GetValueShape(producer->result(0)));
  // auto op_shape = ::common::vectorize<int64_t>(GetFirstInputShape(producer));
  auto op_size = std::accumulate(
      op_shape.begin(), op_shape.end(), 1, std::multiplies<int>());
  auto reduce_size = std::accumulate(
      reduce_shape.begin(), reduce_shape.end(), 1, std::multiplies<int>());

  // is not same size with reduce size.
  if (op_size != reduce_size) {
    return false;
  }
  // check without last axis in reduce.
  if (WithoutLastDimInReduce(reduce_shape, reduce_axes)) {
    return false;
  }

  int successive_reduce_dimension = reduce_shape.at(reduce_axes.back());
  for (int idx = reduce_axes.size() - 2; idx >= 0; --idx) {
    if (reduce_axes[idx] == reduce_axes[idx + 1] - 1) {
      successive_reduce_dimension *= reduce_shape[reduce_axes[idx]];
      continue;
    }
    break;
  }

  // helper->target_ == cinn::common::DefaultNVGPUTarget()
  // successive_reduce_dimension <= helper->target_.max_num_threads()
  // TODO(phlrain): support is_gpu_target and max_thread
  bool is_gpu_target = true;
  int max_thread = 32 * 1024;
  return is_gpu_target
             ? (successive_reduce_dimension <= max_thread ? true : false)
             : true;
}

inline bool horizontal_or_can_inline(
    ::pir::Operation* producer,
    const std::shared_ptr<Group>& consumer,
    ::pir::ShapeConstraintIRAnalysis* shape_analysis) {
  // horizontal relation.
  if (is_horizontal_relation(producer, consumer)) {
    if (is_same_size(producer, consumer, shape_analysis)) {
      return true;
    } else {
      // if do broadcast, check can compute inline.
      // return helper->output_ops_set_.count(producer) == 0;
      // TODO(phlrain): support output op set check
      return false;
    }
  }

  // vertical relation: 1.can compute inline
  if (producer->result(0).use_count() == 1) {
    return true;
  }
  // if (helper->GetNodeData(producer)->outlinks().size() == 1 &&
  //     helper->output_ops_set_.count(producer) == 0) {
  //   return true;
  // }

  // link to same op.
  // auto& out_links = helper->GetNodeData(producer)->outlinks();
  // for (auto link : out_links) {
  //   if ((*out_links.begin())->sink() != link->sink()) {
  //     return false;
  //   }
  // }

  // return helper->output_ops_set_.count(producer) == 0;

  return false;
}

inline bool horizontal_with_same_size(
    ::pir::Operation* producer,
    const std::shared_ptr<Group>& consumer,
    ::pir::ShapeConstraintIRAnalysis* shape_analysis) {
  return is_horizontal_relation(producer, consumer) &&
         is_same_size(producer, consumer, shape_analysis);
}

inline std::vector<int64_t> GetBroadcastAxes(
    ::pir::Operation* bcast_op,
    ::pir::ShapeConstraintIRAnalysis* shape_analysis) {  // NOLINT
  if (bcast_op->isa<cinn::dialect::BroadcastOp>()) {
    return GetVectorAttr(bcast_op, "broadcast_axes");
  } else if (bcast_op->isa<paddle::dialect::ExpandOp>()) {
    const auto& input_shape =
        shape_analysis->GetShapeOrDataForValue(bcast_op->operand_source(0))
            .shape();
    const auto& output_shape =
        shape_analysis->GetShapeOrDataForValue(bcast_op->result(0)).shape();
    std::vector<int64_t> broadcast_axes(input_shape.size(), 0);
    size_t index_gap = output_shape.size() - input_shape.size();
    for (size_t i = 0; i < input_shape.size(); ++i) {
      broadcast_axes[i] = i + index_gap;
    }
    return broadcast_axes;
  } else {
    IR_THROW("Not support broadcast op: %s", bcast_op->name());
  }
}

inline bool reduce_fuse_broadcast(
    ::pir::Operation* producer,
    const std::shared_ptr<Group>& consumer,
    ::pir::ShapeConstraintIRAnalysis* shape_analysis) {
  if (is_horizontal_relation(producer, consumer)) {
    if (is_same_size(producer, consumer, shape_analysis)) {
      return true;
    }
    return false;
  }

  // if (helper->target_ != cinn::common::DefaultNVGPUTarget()) {
  //   return true;
  // }

  const auto& rinput_shape =
      shape_analysis->GetShapeOrDataForValue(producer->operand_source(0))
          .shape();
  auto reduce_axes = GetVectorAttr(producer, "axis");
  auto keepdim = producer->attributes()
                     .at("keepdim")
                     .dyn_cast<::pir::BoolAttribute>()
                     .data();
  for (auto& axis : reduce_axes) {
    if (axis < 0) {
      axis += rinput_shape.size();
    }
  }

  auto find_reducer =
      [&](::pir::Operation* op,
          ::pir::Operation* reducer,
          const std::unordered_set<::pir::Operation*>& ops_set) {
        std::queue<::pir::Operation*> candidates;
        candidates.push(op);

        while (!candidates.empty()) {
          auto candidate = candidates.front();
          candidates.pop();

          for (size_t i = 0; i < candidate->num_operands(); ++i) {
            auto producer = candidate->operand_source(i).defining_op();
            if (producer == reducer) {
              return true;
            }

            if (ops_set.count(producer)) {
              candidates.push(producer);
            }
          }
        }

        return false;
      };

  const auto& routput_shape = GetValueShape(producer->result(0));
  for (auto op : consumer->ops_set) {
    if (hlir::framework::pir::CompatibleInfo::OpKind(*op) !=
        OpPatternKind::kBroadcast) {
      continue;
    }

    if (!find_reducer(op, producer, consumer->ops_set)) {
      continue;
    }
    const auto& broadcast_shape =
        shape_analysis->GetShapeOrDataForValue(op->result(0)).shape();
    auto broadcast_axes = GetBroadcastAxes(op, shape_analysis);

    for (auto& axis : broadcast_axes) {
      if (axis < 0) {
        axis += broadcast_shape.size();
      }
    }

    if (rinput_shape != broadcast_shape) {
      return false;
    }
    // if keep dim = true.
    if (keepdim) {
      continue;
    } else {
      // if routput_shape = [1]
      if (routput_shape.size() == 1 && routput_shape[0] == 1) {
        continue;
      }
      // check [reduce_axes, axes] = {0, 1, 2, 3, 4, 5, 6, ...}
      for (size_t idx = 0; idx < rinput_shape.size(); ++idx) {
        // note: !x ^ y == (!x) ^ y == !(x ^ y)
        if ((std::find(broadcast_axes.begin(), broadcast_axes.end(), idx) !=
             broadcast_axes.end()) ^
            std::find(reduce_axes.begin(), reduce_axes.end(), idx) ==
                reduce_axes.end()) {
          return false;
        }
      }
      continue;
    }
    return false;
  }
  return true;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
