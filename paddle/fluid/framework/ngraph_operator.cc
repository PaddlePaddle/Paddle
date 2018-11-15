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

#ifdef PADDLE_WITH_NGRAPH
#include <glog/logging.h>

#include <algorithm>
#include <map>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/ngraph_operator.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace framework {

static std::map<proto::VarType::Type, ngraph::element::Type> pd2ng_type_map = {
    {proto::VarType::FP32, ngraph::element::f32},
    {proto::VarType::FP64, ngraph::element::f64},
    {proto::VarType::INT32, ngraph::element::i32},
    {proto::VarType::INT64, ngraph::element::i64},
    {proto::VarType::BOOL, ngraph::element::boolean},
};

typedef enum {                /* nGraph support state on ops          */
               FULL_TRAIN,    /* Support full ops for train           */
               PARTIAL_TRAIN, /* Support partial ops for train        */
               FULL_TEST,     /* Support full list of ops for test    */
               PARTIAL_TEST   /* Support partial list of ops for test */
} op_state;

class NgraphOperator {
 public:
  explicit NgraphOperator(const Scope& scope, const platform::Place& place,
                          const std::vector<std::shared_ptr<OperatorBase>>& ops,
                          const std::unordered_map<
                              std::string, ngraph::element::Type>& var_type_map,
                          const std::unordered_set<std::string>& persist,
                          const std::unordered_set<std::string>& fetches,
                          const std::unordered_set<std::string>& post_op_inputs,
                          op_state ng_op_state)
      : scope_(scope),
        place_(place),
        fused_ops_(ops),
        var_type_map_(var_type_map),
        persistables_(persist),
        fetches_(fetches),
        post_op_inputs_(post_op_inputs),
        ng_op_state_(ng_op_state) {}

  void Run(const Scope& scope, const platform::Place& place) const;

 private:
  static std::unordered_map<std::string, std::shared_ptr<ngraph::Function>>
      func_cache;
  const Scope& scope_;
  const platform::Place& place_;
  std::vector<std::shared_ptr<OperatorBase>> fused_ops_;
  std::unordered_map<std::string, ngraph::element::Type> var_type_map_;
  std::unordered_set<std::string> persistables_;
  std::unordered_set<std::string> fetches_;
  std::unordered_set<std::string> post_op_inputs_;
  op_state ng_op_state_;
};

std::vector<std::vector<std::vector<std::unique_ptr<OperatorBase>>::iterator>>
FusedOperator::FusedOpIntervals(
    std::vector<std::unique_ptr<paddle::framework::OperatorBase>>* ops) {
  std::vector<std::vector<std::vector<std::unique_ptr<OperatorBase>>::iterator>>
      intervals;
  if (ops->empty()) {
    return intervals;
  }
  size_t size = ops->size();
  size_t left = 0;
  while (left < size && ops.at(left)->Type() != kFeedOpType) {
    ++left;
  }
  if (left == size) {
    return intervals;
  }
  while (left < size && ops->at(left)->Type() == kFeedOpType) {
    ++left;
  }

  size_t right = left;
  while (right < size && ops->at(right)->Type() != kFetchOpType) {
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
    if (paddle::framework::NgraphBridge::NG_NODE_MAP.find(op_type) ==
        paddle::framework::NgraphBridge::NG_NODE_MAP.end()) {
      ++pivot;
    } else {
      size_t start = pivot, end = start;
      while (pivot < right &&
             (paddle::framework::NgraphBridge::NG_NODE_MAP.find(
                  ops.at(pivot)->Type()) !=
              paddle::framework::NgraphBridge::NG_NODE_MAP.end())) {
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

FusedOperator::FusedOperator(
    const ProgramDesc& prog, size_t block_id,
    std::vector<std::unique_ptr<OperatorBase>>::iterator start,
    std::vector<std::unique_ptr<OperatorBase>>::iterator end,
    const std::string& type, const VariableNameMap& inputs,
    const VariableNameMap& outputs, const AttributeMap& attrs)
    : OperatorBase(type, inputs, outputs, attrs), pdesc(prog), block(block_id) {
  for (std::vector<std::unique_ptr<OperatorBase>>::iterator it = start;
       it != end; ++it) {
    fused_ops_.push_back(std::move(*it));
  }

  for (std::vector<std::unique_ptr<OperatorBase>>::iterator it = end;
       (*it)->Type() != kFetchOpType; ++it) {
    for (auto& var_name_item : (*it)->Inputs()) {
      for (auto& var_name : var_name_item.second) {
        post_op_inputs_.insert(var_name);
      }
    }
  }

  if ((*(start - 1))->Type() == kFeedOpType && (*end)->Type() == kFetchOpType) {
    is_complete = true;
  }

  Process();
}

void FusedOperator::Process() {
  auto& bdesc = pdesc_.Block(block_);
  for (auto& var : bdesc.AllVars()) {
    if (!(var->GetType() == proto::VarType::SELECTED_ROWS ||
          var->GetType() == proto::VarType::LOD_TENSOR ||
          var->GetType() == proto::VarType::LOD_TENSOR_ARRAY)) {
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
    if (op->Type() == kFetchOpType) {
      std::string fetch_target_name = op->Input("X")[0];
      fetches_.insert(fetch_target_name);
    }
  }
}

void FusedOperator::RunImpl(const Scope& scope,
                            const platform::Place& place) const {
  op_state ng_op_state = PARTIAL_TEST;
  auto& bdesc = pdesc_.Block(block_);
  for (auto* op : bdesc.AllOps()) {
    if (op->Type().find("_grad") != std::string::npos) {
      ng_op_state = PARTIAL_TRAIN;
      break;
    }
  }

  if (is_full) {
    ng_op_state = ng_op_state == PARTIAL_TEST ? FULL_TEST : FULL_TRAIN;
  }

  NgraphOperator ngraph_op(scope, place, fused_ops_, var_type_map_,
                           persistables_, fetches_, post_op_inputs_,
                           ng_op_state);
  ngraph_op.Run(scope, place);
}

}  // namespace framework
}  // namespace paddle
#endif
