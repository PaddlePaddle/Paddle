// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_group.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"

namespace cinn {
namespace dialect {
namespace ir {

using OpGroupPtr = ir::OpGroup;
using OpGroupList = std::vector<OpGroupPtr>;

static cinn::dialect::ir::OpNode GetMasterNode(const OpGroupPtr& op_group) {
  std::vector<cinn::dialect::ir::OpNode> master_nodes;
  op_group.WalkOpNodes([&](const cinn::dialect::ir::OpNode& op) {
    if (op.kind() == OpPatternKind::kReduction) {
      master_nodes.push_back(op);
    }
  });
  if (!master_nodes.empty()) {
    return master_nodes.front();
  }

  op_group.WalkOpNodes(
      [&](const cinn::dialect::ir::OpNode& op) { master_nodes.push_back(op); });
  return master_nodes.back();
}

static bool IsSameSize(const OpGroupPtr& src, const OpGroupPtr& dst) {
  cinn::dialect::ir::OpNode src_master_node = GetMasterNode(src);
  cinn::dialect::ir::OpNode dst_master_node = GetMasterNode(dst);

  auto size_0 = src_master_node.outputs()[0].shape();
  auto size_1 = dst_master_node.outputs()[0].shape();

  return ::common::product(size_0) == ::common::product(size_1);
}

static std::unordered_set<cinn::dialect::ir::OpNode> GetInputOps(
    const OpGroupPtr& op_group) {
  std::unordered_set<OpNode> ops_set;
  op_group.WalkOpNodes([&ops_set](const cinn::dialect::ir::OpNode& op_node) {
    ops_set.insert(op_node);
  });

  std::unordered_set<cinn::dialect::ir::OpNode> input_ops;
  op_group.WalkOpNodes([&](const cinn::dialect::ir::OpNode& op) {
    const auto& input_tensors = op.inputs();
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      if (!ops_set.count(input_tensors[i].producer())) {
        input_ops.insert(input_tensors[i].producer());
      }
    }
  });
  return input_ops;
}

static std::unordered_set<cinn::dialect::ir::OpNode> GetOutputOps(
    const OpGroupPtr& op_group) {
  std::unordered_set<OpNode> ops_set;
  op_group.WalkOpNodes([&ops_set](const cinn::dialect::ir::OpNode& op_node) {
    ops_set.insert(op_node);
  });
  std::unordered_set<cinn::dialect::ir::OpNode> output_ops;
  op_group.WalkOpNodes([&](const cinn::dialect::ir::OpNode& op) {
    const auto& output_tensors = op.outputs();
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      auto& consumers = output_tensors[i].consumers();
      for (auto it = consumers.begin(); it != consumers.end(); ++it) {
        if (!ops_set.count(*it)) {
          output_ops.insert(*it);
          break;
        }
      }
    }
  });
  return output_ops;
}

// limit the group args number to less equal 512, as args stack size is 4K.
static bool limit_args(const OpGroupPtr& first, const OpGroupPtr& second) {
  std::unordered_set<cinn::dialect::ir::OpNode> args;
  for (auto& group : {first, second}) {
    for (const auto& node : GetInputOps(group)) {
      args.insert(node);
    }
    for (const auto& node : GetOutputOps(group)) {
      args.insert(node);
    }
  }

  if (args.size() > 512) {
    return false;
  } else {
    return true;
  }
}

bool WithoutLastDimInReduce(const phi::DDim& inshape,
                            const std::vector<int64_t>& axes) {
  // if last axis is in reduce.
  if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
      std::find(axes.begin(), axes.end(), -1) != axes.end()) {
    return false;
  }

  int sum_last_axes = 1;
  for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
    sum_last_axes *= inshape[idx];
  }

  if (sum_last_axes > 1) {
    return true;
  } else {
    return false;
  }
}

static int GetSharedSize(const cinn::dialect::ir::OpNode& op_node) {
  const auto& inshape = op_node.inputs()[0].shape();
  const auto& axes = op_node.GetAttr<std::vector<int64_t>>("dim");

  if (WithoutLastDimInReduce(inshape, axes)) {
    int lane = 1;
    for (int idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      lane = inshape[idx];
    }
    // int max_num_threads =
    // cinn::common::DefaultDeviceTarget().max_num_threads();
    int max_num_threads = 1000;
    if (lane > max_num_threads / 2) {
      return 0;
    }
    int index = axes.size() - 1;
    for (; index >= 0; --index) {
      if (static_cast<size_t>(index + 1) < axes.size() &&
          axes[index] != axes[index + 1] - 1) {
        break;
      }
      lane *= inshape[axes[index]];
      if (lane > max_num_threads / 2) {
        break;
      }
    }
    // if lane > (max_num_threads / 2),the loop break from lane >
    // max_num_threads / 2.
    int axis = lane > (max_num_threads / 2) ? axes[index] : axes[index + 1];
    if (lane <= max_num_threads) {
      return lane * sizeof(float);
    } else {
      int prefix = inshape[axis];
      int tail = lane / prefix;
      for (int idx = max_num_threads / tail;
           idx > ((max_num_threads / 2) / tail);
           --idx) {
        if (prefix % idx == 0) {
          return idx * tail * sizeof(float);
        }
      }
      int num = max_num_threads / tail;
      return num * tail * sizeof(float);
    }
  }
  return 0;
}

static bool ReduceFuseReduce1(const OpGroupPtr& first,
                              const OpGroupPtr& second) {
  // return false;
  // if (!limit_args(first, second)) {
  //   return false;
  // }
  std::unique_ptr<cinn::dialect::ir::OpNode> reducer_0 = nullptr;
  for (auto op : first.GetGroup()->CollectOps()) {
    if (hlir::framework::pir::CompatibleInfo::OpKind(*op) ==
        OpPatternKind::kReduction) {
      reducer_0.reset(new cinn::dialect::ir::OpNode(op));
      break;
    }
  }

  CHECK(reducer_0) << "Can't find reduce op in group " << first.group_id();

  std::unique_ptr<cinn::dialect::ir::OpNode> reducer_1 = nullptr;
  second.WalkOpNodes([&](const cinn::dialect::ir::OpNode& op) {
    if (!reducer_1 && op.kind() == OpPatternKind::kReduction) {
      reducer_1.reset(new cinn::dialect::ir::OpNode(op));
    }
  });

  CHECK(reducer_1) << "Can't find reduce op in group " << second.group_id();

  // check reduce has same input shape and output shape
  const auto& reducer_0_input_shape = reducer_0->inputs()[0].shape();
  const auto& reducer_0_output_shape = reducer_0->outputs()[0].shape();

  const auto& reducer_1_input_shape = reducer_1->inputs()[0].shape();
  const auto& reducer_1_output_shape = reducer_1->outputs()[0].shape();

  auto reducer_0_reduce_dim = reducer_0->GetAttr<std::vector<int64_t>>("dim");
  auto reducer_1_reduce_dim = reducer_1->GetAttr<std::vector<int64_t>>("dim");

  for (auto& dim : reducer_0_reduce_dim) {
    // if dim = -1, set as shape.size() - 1
    if (dim == -1) {
      dim = reducer_0_reduce_dim.size() - 1;
    }
  }

  for (auto& dim : reducer_1_reduce_dim) {
    // if dim = -1,  set as shape.size() - 1
    if (dim == -1) {
      dim = reducer_1_reduce_dim.size() - 1;
    }
  }

  // check shape is same
  if (reducer_0_input_shape == reducer_1_input_shape &&
      reducer_0_output_shape == reducer_1_output_shape &&
      reducer_0_reduce_dim == reducer_1_reduce_dim) {
    auto shared_size = 0;
    for (auto& fusion_group : {first, second}) {
      fusion_group.WalkOpNodes([&](const cinn::dialect::ir::OpNode& op) {
        if (op.kind() == OpPatternKind::kReduction) {
          shared_size += GetSharedSize(op);
        }
      });
    }

#define MAX_AVAILABLE_SHREAD 32 * 1024
    if (shared_size > MAX_AVAILABLE_SHREAD) {
      return false;
    }
#undef MAX_AVAILABLE_SHREAD
    return true;
  }

  if (WithoutLastDimInReduce(reducer_0_input_shape, reducer_0_reduce_dim) &&
      WithoutLastDimInReduce(reducer_1_input_shape, reducer_1_reduce_dim) &&
      reducer_0_output_shape == reducer_1_output_shape &&
      reducer_0_reduce_dim == reducer_1_reduce_dim) {
    auto shared_size = 0;
    for (auto& fusion_group : {first, second}) {
      fusion_group.WalkOpNodes([&](const cinn::dialect::ir::OpNode& op) {
        if (op.kind() == OpPatternKind::kReduction) {
          shared_size += GetSharedSize(op);
        }
      });
    }

#define MAX_AVAILABLE_SHREAD 32 * 1024
    if (shared_size > MAX_AVAILABLE_SHREAD) {
      return false;
    }
#undef MAX_AVAILABLE_SHREAD
    return true;
  }

  return false;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
