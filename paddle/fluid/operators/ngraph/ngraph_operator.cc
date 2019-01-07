/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>

#include <algorithm>
#include <map>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/ngraph/ngraph_bridge.h"
#include "paddle/fluid/operators/ngraph/ngraph_engine.h"
#include "paddle/fluid/operators/ngraph/ngraph_operator.h"

#include "ngraph/ngraph.hpp"

namespace paddle {
namespace operators {

static std::map<framework::proto::VarType::Type, ngraph::element::Type>
    pd2ng_type_map = {
        {framework::proto::VarType::FP32, ngraph::element::f32},
        {framework::proto::VarType::FP64, ngraph::element::f64},
        {framework::proto::VarType::INT32, ngraph::element::i32},
        {framework::proto::VarType::INT64, ngraph::element::i64},
        {framework::proto::VarType::BOOL, ngraph::element::boolean},
};

std::vector<std::vector<
    std::vector<std::unique_ptr<framework::OperatorBase>>::iterator>>
NgraphOperator::NgraphOpIntervals(
    std::vector<std::unique_ptr<framework::OperatorBase>>* ops) {
  std::vector<std::vector<
      std::vector<std::unique_ptr<framework::OperatorBase>>::iterator>>
      intervals;
  if (ops->empty()) {
    return intervals;
  }
  size_t size = ops->size();
  size_t left = 0;
  while (left < size && ops->at(left)->Type() != framework::kFeedOpType) {
    ++left;
  }
  if (left == size) {
    return intervals;
  }
  while (left < size && ops->at(left)->Type() == framework::kFeedOpType) {
    ++left;
  }

  size_t right = left;
  while (right < size && ops->at(right)->Type() != framework::kFetchOpType) {
    ++right;
  }
  if (right == size) {
    return intervals;
  }
  if (left >= right) return intervals;

  // (left, right - 1) represents indices between feed and fetch
  size_t pivot = left;
  while (pivot < right) {
    auto op_type = ops->at(pivot)->Type();
    if (operators::NgraphBridge::NG_NODE_MAP.find(op_type) ==
        operators::NgraphBridge::NG_NODE_MAP.end()) {
      ++pivot;
    } else {
      size_t start = pivot, end = start;
      while (pivot < right && (operators::NgraphBridge::NG_NODE_MAP.find(
                                   ops->at(pivot)->Type()) !=
                               operators::NgraphBridge::NG_NODE_MAP.end())) {
        ++pivot;
        ++end;
      }
      std::vector<std::vector<std::unique_ptr<OperatorBase>>::iterator>
          interval = {ops->begin() + start, ops->begin() + end};
      intervals.push_back(interval);
    }
  }  // end while

  return intervals;
}

NgraphOperator::NgraphOperator(
    const framework::ProgramDesc& prog, size_t block_id,
    std::vector<std::unique_ptr<framework::OperatorBase>>::iterator start,
    std::vector<std::unique_ptr<framework::OperatorBase>>::iterator end,
    const std::string& type, const framework::VariableNameMap& inputs,
    const framework::VariableNameMap& outputs,
    const framework::AttributeMap& attrs)
    : OperatorBase(type, inputs, outputs, attrs),
      pdesc_(prog),
      block_(block_id) {
  for (std::vector<std::unique_ptr<OperatorBase>>::iterator it = start;
       it != end; ++it) {
    fused_ops_.push_back(std::move(*it));
  }

  for (std::vector<std::unique_ptr<OperatorBase>>::iterator it = end;
       (*it)->Type() != framework::kFetchOpType; ++it) {
    for (auto& var_name_item : (*it)->Inputs()) {
      for (auto& var_name : var_name_item.second) {
        post_op_inputs_.insert(var_name);
      }
    }
  }

  if ((*(start - 1))->Type() == framework::kFeedOpType &&
      (*end)->Type() == framework::kFetchOpType) {
    is_full_ = true;
  }

  Process();
}

void NgraphOperator::Process() {
  auto& bdesc = pdesc_.Block(block_);
  for (auto& var : bdesc.AllVars()) {
    if (!(var->GetType() == framework::proto::VarType::SELECTED_ROWS ||
          var->GetType() == framework::proto::VarType::LOD_TENSOR ||
          var->GetType() == framework::proto::VarType::LOD_TENSOR_ARRAY)) {
      continue;
    }

    auto var_name = var->Name();
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }

    if (var_name != "fetch" && var_name != "feed") {
      auto pd_type = var->GetDataType();
      if (pd2ng_type_map.find(pd_type) == pd2ng_type_map.end()) {
        PADDLE_THROW("Data type of var %s not found in pd2ng_type_map",
                     var_name);
      }
      var_type_map_[var_name] = pd2ng_type_map[pd_type];
    }

    if (var->Persistable()) {
      persistables_.insert(var->Name());
    }
  }

  for (auto* op : bdesc.AllOps()) {
    if (op->Type() == framework::kFetchOpType) {
      std::string fetch_target_name = op->Input("X")[0];
      fetches_.insert(fetch_target_name);
    }
  }
}

void NgraphOperator::RunImpl(const framework::Scope& scope,
                             const platform::Place& place) const {
  op_state ng_op_state = PARTIAL_TEST;
  auto& bdesc = pdesc_.Block(block_);
  for (auto* op : bdesc.AllOps()) {
    if (op->Type().find("_grad") != std::string::npos) {
      ng_op_state = PARTIAL_TRAIN;
      break;
    }
  }

  if (is_full_) {
    ng_op_state = ng_op_state == PARTIAL_TEST ? FULL_TEST : FULL_TRAIN;
  }

  NgraphEngine ngraph_engine(scope, place, fused_ops_, var_type_map_,
                             persistables_, fetches_, post_op_inputs_,
                             ng_op_state);
  ngraph_engine.Run(scope, place);
}

}  // namespace operators
}  // namespace paddle
