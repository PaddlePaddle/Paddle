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
#include "paddle/fluid/framework/ngraph_bridge.h"
#include "paddle/fluid/framework/ngraph_operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/var_type.h"

#include "ngraph/ngraph.hpp"

namespace paddle {
namespace framework {

static ngraph::Shape Ddim2Shape(const DDim& dims) {
  ngraph::Shape sp;
  for (int i = 0; i < dims.size(); ++i) {
    int k = dims[i];
    k = k == 0 ? 1 : k;
    sp.push_back(k);
  }
  return sp;
}

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

// perform graph build through bridge and execute computation
class NgraphEngine {
 public:
  explicit NgraphEngine(const Scope& scope, const platform::Place& place,
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
        ng_op_state_(ng_op_state) {
    var_in_node_map_ = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>();

    var_node_map_ = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>();

    BuildNgIO();

    GetNgFunction();
  }

  void Run(const Scope& scope, const platform::Place& place) const;

 private:
  static std::unordered_map<std::string, std::shared_ptr<ngraph::Function>>
      func_cache_;
  const Scope& scope_;
  const platform::Place& place_;
  std::vector<std::shared_ptr<OperatorBase>> fused_ops_;
  std::unordered_map<std::string, ngraph::element::Type> var_type_map_;
  std::unordered_set<std::string> persistables_;
  std::unordered_set<std::string> fetches_;
  std::unordered_set<std::string> post_op_inputs_;
  op_state ng_op_state_;

  // ngraph backend eg. CPU
  static std::shared_ptr<ngraph::runtime::Backend> backend_;
  // ngraph function to call and execute
  std::shared_ptr<ngraph::Function> ngraph_function_;
  // var_name of inputs
  std::vector<std::string> var_in_;
  // var_name of outputs from  fetch in order
  std::vector<std::string> var_out_;
  // map input vars to nodes
  std::shared_ptr<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
      var_in_node_map_;
  // map each var name with a ngraph node
  std::shared_ptr<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
      var_node_map_;
  // cache key to check if function is cached
  std::shared_ptr<std::string> GetCacheKey();
  // get ngraph input and define ngraph input parameters
  void GetNgInputShape(std::shared_ptr<OperatorBase> op);
  // Call ngraph bridge to map ops
  void BuildNgNodes();
  // get the ngraph input and output var list
  void BuildNgIO();
  // build ngraph function call
  void BuildNgFunction();
  // Check cache for ngraph function or otherwise build the function
  void GetNgFunction();
};

std::vector<std::vector<std::vector<std::unique_ptr<OperatorBase>>::iterator>>
NgraphOperator::NgraphOpIntervals(
    std::vector<std::unique_ptr<paddle::framework::OperatorBase>>* ops) {
  std::vector<std::vector<std::vector<std::unique_ptr<OperatorBase>>::iterator>>
      intervals;
  if (ops->empty()) {
    return intervals;
  }
  size_t size = ops->size();
  size_t left = 0;
  while (left < size && ops->at(left)->Type() != kFeedOpType) {
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
                  ops->at(pivot)->Type()) !=
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

NgraphOperator::NgraphOperator(
    const ProgramDesc& prog, size_t block_id,
    std::vector<std::unique_ptr<OperatorBase>>::iterator start,
    std::vector<std::unique_ptr<OperatorBase>>::iterator end,
    const std::string& type, const VariableNameMap& inputs,
    const VariableNameMap& outputs, const AttributeMap& attrs)
    : OperatorBase(type, inputs, outputs, attrs),
      pdesc_(prog),
      block_(block_id) {
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
    is_full_ = true;
  }

  Process();
}

void NgraphOperator::Process() {
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

void NgraphOperator::RunImpl(const Scope& scope,
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

std::unordered_map<std::string, std::shared_ptr<ngraph::Function>>
    NgraphEngine::func_cache_ = {};

std::shared_ptr<ngraph::runtime::Backend> NgraphEngine::backend_ =
    ngraph::runtime::Backend::create("CPU");

void NgraphEngine::GetNgInputShape(std::shared_ptr<OperatorBase> op) {
  RuntimeContext ctx(op->Inputs(), op->Outputs(), scope_);
  op->RuntimeInferShape(scope_, place_, ctx);
  for (auto& var_name_item : op->Inputs()) {
    for (auto& var_name : var_name_item.second) {
      auto* var = scope_.FindVar(var_name);
      if (var && var->IsType<LoDTensor>()) {
        auto* tensor_pd = GetLoDTensorOrSelectedRowsValueFromVar(*var);
        auto sp = Ddim2Shape(tensor_pd->dims());
        if (std::find(var_in_.begin(), var_in_.end(), var_name) !=
            var_in_.end()) {
          if (var_node_map_->find(var_name) == var_node_map_->end()) {
            auto ng_type = var_type_map_.at(var_name);
            auto prm =
                std::make_shared<ngraph::op::Parameter>(ng_type, sp, true);
            (*var_node_map_)[var_name] = prm;
            (*var_in_node_map_)[var_name] = prm;
          }
        }
      }
    }
  }
}

void NgraphEngine::BuildNgNodes() {
  for (auto& var_name : var_out_) {
    if (var_node_map_->find(var_name) == var_node_map_->end()) {
      auto* var = scope_.FindVar(var_name);
      if (var && var->IsType<LoDTensor>()) {
        auto* tensor_pd = GetLoDTensorOrSelectedRowsValueFromVar(*var);
        auto& ddim = tensor_pd->dims();
        auto ng_shape = Ddim2Shape(ddim);
        auto ng_type = var_type_map_.at(var_name);
        auto prm =
            std::make_shared<ngraph::op::Parameter>(ng_type, ng_shape, true);
        (*var_node_map_)[var_name] = prm;
      }
    }
  }

  paddle::framework::NgraphBridge ngb(var_node_map_);
  for (auto& op : fused_ops_) {
    ngb.BuildNgNode(op);
  }
}

void NgraphEngine::BuildNgIO() {
  std::unordered_set<std::string> inputs;
  std::unordered_set<std::string> outputs;

  for (auto& op : fused_ops_) {
    for (auto& var_name_item : op->Inputs()) {
      for (auto& var_name : var_name_item.second) {
        inputs.insert(var_name);
        const bool is_output = outputs.find(var_name) != outputs.end();
        if (!is_output &&
            std::find(var_in_.begin(), var_in_.end(), var_name) ==
                var_in_.end()) {
          // fill var_in here to keep lhs and rhs order
          var_in_.push_back(var_name);
        }
      }
    }

    if (op->Type() != "fill_constant") {
      GetNgInputShape(op);
    }

    for (auto& var_name_item : op->Outputs()) {
      PADDLE_ENFORCE_LE(var_name_item.second.size(), 1,
                        "op %s has more than 1 output - Not handling yet",
                        op->Type());
      for (auto& var_name : var_name_item.second) {
        outputs.insert(var_name);
      }
    }
  }

  // var_out.clear();
  for (auto& op : fused_ops_) {
    for (auto& var_name_item : op->Outputs()) {
      PADDLE_ENFORCE_LE(var_name_item.second.size(), 1,
                        "op %s has more than 1 output - Not handling yet",
                        op->Type());
      for (auto& var_name : var_name_item.second) {
        switch (ng_op_state_) {
          case PARTIAL_TEST:
            if (post_op_inputs_.find(var_name) != post_op_inputs_.end() ||
                fetches_.find(var_name) != fetches_.end()) {
              var_out_.push_back(var_name);
            }
            break;
          case FULL_TEST:
            if (fetches_.find(var_name) != fetches_.end()) {
              var_out_.push_back(var_name);
            }
            break;
          case PARTIAL_TRAIN:
            if (fetches_.find(var_name) != fetches_.end() ||
                post_op_inputs_.find(var_name) != post_op_inputs_.end() ||
                persistables_.find(var_name) != persistables_.end()) {
              var_out_.push_back(var_name);
            }
            break;
          case FULL_TRAIN:
            if (fetches_.find(var_name) != fetches_.end() ||
                persistables_.find(var_name) != persistables_.end()) {
              var_out_.push_back(var_name);
            }
            break;
          default:
            var_out_.push_back(var_name);
        }
      }
    }
  }
}

void NgraphEngine::BuildNgFunction() {
  BuildNgNodes();
  ngraph_function_ = nullptr;
  ngraph::NodeVector func_outputs;
  ngraph::ParameterVector func_inputs;

  for (auto& vo : var_out_) {
    func_outputs.push_back(var_node_map_->at(vo));
  }

  for (auto& vi : var_in_) {
    std::shared_ptr<ngraph::op::Parameter> prm =
        std::dynamic_pointer_cast<ngraph::op::Parameter>(
            var_in_node_map_->at(vi));
    func_inputs.push_back(prm);
  }

  ngraph_function_ =
      std::make_shared<ngraph::Function>(func_outputs, func_inputs);
}

std::shared_ptr<std::string> NgraphEngine::GetCacheKey() {
  auto cache_key = std::make_shared<std::string>("");
  *cache_key += std::to_string(fused_ops_.size());
  for (auto& op : fused_ops_) {
    *cache_key += op->Type();
  }
  for (auto& var_name : var_in_) {
    auto shape = var_node_map_->at(var_name)->get_shape();
    *cache_key += var_name;
    *cache_key += var_type_map_.at(var_name).c_type_string();
    for (size_t i = 0; i < shape.size(); ++i) {
      *cache_key += std::to_string(shape.at(i));
    }
  }

  for (auto& var_name : var_out_) {
    auto* var = scope_.FindVar(var_name);
    if (var && var->IsType<LoDTensor>()) {
      auto* tensor_pd = GetLoDTensorOrSelectedRowsValueFromVar(*var);
      auto& ddim = tensor_pd->dims();
      for (int i = 0; i < ddim.size(); ++i) {
        *cache_key += std::to_string(ddim[i]);
      }
    }
  }
  return cache_key;
}

void NgraphEngine::GetNgFunction() {
  bool cache_on = true;
  if (cache_on) {
    std::string cache_key_val = *GetCacheKey();
    if (func_cache_.find(cache_key_val) != func_cache_.end()) {
      ngraph_function_ = func_cache_.at(cache_key_val);
    } else {
      BuildNgFunction();
      func_cache_[cache_key_val] = ngraph_function_;
    }
  } else {
    BuildNgFunction();
  }
}

void NgraphEngine::Run(const Scope& scope, const platform::Place& place) const {
  std::vector<std::shared_ptr<ngraph::runtime::Tensor>> t_in;
  std::vector<std::shared_ptr<ngraph::runtime::Tensor>> t_out;

  for (size_t i = 0; i < var_in_.size(); ++i) {
    auto vi = var_in_.at(i);
    auto sp = var_node_map_->at(vi)->get_shape();
    std::shared_ptr<ngraph::runtime::Tensor> ti;
    auto* var = scope.FindVar(vi);
    if (var && var->IsType<LoDTensor>()) {
      auto* tensor_pd = GetLoDTensorOrSelectedRowsValueFromVar(*var);
      PADDLE_ENFORCE(sp == Ddim2Shape(tensor_pd->dims()),
                     "Ensure ngraph tensor layout align with paddle tensor");
      if (tensor_pd->type() == proto::VarType::FP32) {
        const float* arr = tensor_pd->data<float>();
        ti = backend_->create_tensor(ngraph::element::f32, sp,
                                     const_cast<float*>(arr));
      } else if (tensor_pd->type() == proto::VarType::INT32) {
        const int* arr = tensor_pd->data<int>();
        ti = backend_->create_tensor(ngraph::element::i32, sp,
                                     const_cast<int*>(arr));
      } else if (tensor_pd->type() == proto::VarType::INT64) {
        const int64_t* arr = tensor_pd->data<int64_t>();
        ti = backend_->create_tensor(ngraph::element::i64, sp,
                                     const_cast<int64_t*>(arr));
      } else if (tensor_pd->type() == proto::VarType::FP64) {
        const double* arr = tensor_pd->data<double>();
        ti = backend_->create_tensor(ngraph::element::f64, sp,
                                     const_cast<double*>(arr));
      } else if (tensor_pd->type() == proto::VarType::BOOL) {
        const bool* arr = tensor_pd->data<bool>();
        ti = backend_->create_tensor(ngraph::element::boolean, sp,
                                     const_cast<bool*>(arr));
      } else {
        PADDLE_THROW("Data type not handling for var %s", vi);
      }
    } else {
      PADDLE_THROW("Cannot find var or tensor with var name %s", vi);
    }
    bool is_test = (ng_op_state_ == PARTIAL_TEST || ng_op_state_ == FULL_TEST)
                       ? true
                       : false;
    bool is_persistable =
        (persistables_.find(vi) != persistables_.end()) ? true : false;
    if (is_test && is_persistable) {
      ti->set_stale(false);
    }
    t_in.push_back(ti);
  }

  for (size_t i = 0; i < var_out_.size(); ++i) {
    auto var_name = var_out_[i];
    auto* var = scope.FindVar(var_name);
    std::shared_ptr<ngraph::runtime::Tensor> to;
    if (var && var->IsType<LoDTensor>()) {
      auto* tensor_pd = GetMutableLoDTensorOrSelectedRowsValueFromVar(var);
      auto dd = tensor_pd->dims();
      ngraph::Shape sp = Ddim2Shape(dd);
      auto ng_type = var_type_map_.at(var_name);
      if (ng_type == ngraph::element::f32) {
        auto pd_arr = tensor_pd->mutable_data<float>(place);
        to = backend_->create_tensor(ngraph::element::f32, sp, pd_arr);
      } else if (ng_type == ngraph::element::i64) {
        auto pd_arr = tensor_pd->mutable_data<int64_t>(place);
        to = backend_->create_tensor(ngraph::element::i64, sp, pd_arr);
      } else if (ng_type == ngraph::element::f64) {
        auto pd_arr = tensor_pd->mutable_data<double>(place);
        to = backend_->create_tensor(ngraph::element::f64, sp, pd_arr);
      } else if (ng_type == ngraph::element::boolean) {
        auto pd_arr = tensor_pd->mutable_data<bool>(place);
        to = backend_->create_tensor(ngraph::element::boolean, sp, pd_arr);
      } else {
        PADDLE_THROW("Data type not handled in for var %s", var_name);
      }
      t_out.push_back(to);
    } else {
      PADDLE_THROW("Cannot find var or tensor with var name %s", var_name);
    }
  }

  backend_->call(ngraph_function_, t_out, t_in);
}  // NgraphEngine::RunImpl
}  // namespace framework
}  // namespace paddle
