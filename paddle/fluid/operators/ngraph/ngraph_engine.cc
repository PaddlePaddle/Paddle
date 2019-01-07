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

namespace paddle {
namespace operators {

static ngraph::Shape Ddim2Shape(const framework::DDim& dims) {
  ngraph::Shape sp;
  for (int i = 0; i < dims.size(); ++i) {
    int k = dims[i];
    k = k == 0 ? 1 : k;
    sp.push_back(k);
  }
  return sp;
}

static std::map<framework::proto::VarType::Type, ngraph::element::Type>
    pd2ng_type_map = {
        {framework::proto::VarType::FP32, ngraph::element::f32},
        {framework::proto::VarType::FP64, ngraph::element::f64},
        {framework::proto::VarType::INT32, ngraph::element::i32},
        {framework::proto::VarType::INT64, ngraph::element::i64},
        {framework::proto::VarType::BOOL, ngraph::element::boolean},
};

std::unordered_map<std::string, std::shared_ptr<ngraph::Function>>
    NgraphEngine::func_cache_ = {};

std::shared_ptr<ngraph::runtime::Backend> NgraphEngine::backend_ =
    ngraph::runtime::Backend::create("CPU");

void NgraphEngine::GetNgInputShape(
    std::shared_ptr<framework::OperatorBase> op) {
  framework::RuntimeContext ctx(op->Inputs(), op->Outputs(), scope_);
  op->RuntimeInferShape(scope_, place_, ctx);
  for (auto& var_name_item : op->Inputs()) {
    for (auto& var_name : var_name_item.second) {
      auto* var = scope_.FindVar(var_name);
      if (var && var->IsType<framework::LoDTensor>()) {
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
  for (auto& op : fused_ops_) {
    for (auto& var_name_item : op->Outputs()) {
      for (auto& var_name : var_name_item.second) {
        if (var_node_map_->find(var_name) == var_node_map_->end()) {
          auto* var = scope_.FindVar(var_name);
          if (var && var->IsType<framework::LoDTensor>()) {
            auto* tensor_pd = GetLoDTensorOrSelectedRowsValueFromVar(*var);
            auto& ddim = tensor_pd->dims();
            auto ng_shape = Ddim2Shape(ddim);
            auto ng_type = var_type_map_.at(var_name);
            auto prm = std::make_shared<ngraph::op::Parameter>(ng_type,
                                                               ng_shape, true);
            (*var_node_map_)[var_name] = prm;
          }
        }
      }
    }
  }

  paddle::operators::NgraphBridge ngb(var_node_map_);
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
    if (var && var->IsType<framework::LoDTensor>()) {
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

void NgraphEngine::Run(const framework::Scope& scope,
                       const platform::Place& place) const {
  std::vector<std::shared_ptr<ngraph::runtime::Tensor>> t_in;
  std::vector<std::shared_ptr<ngraph::runtime::Tensor>> t_out;

  for (size_t i = 0; i < var_in_.size(); ++i) {
    auto vi = var_in_.at(i);
    auto sp = var_node_map_->at(vi)->get_shape();
    std::shared_ptr<ngraph::runtime::Tensor> ti;
    auto* var = scope.FindVar(vi);
    if (var && var->IsType<framework::LoDTensor>()) {
      auto* tensor_pd = GetMutableLoDTensorOrSelectedRowsValueFromVar(var);
      PADDLE_ENFORCE(sp == Ddim2Shape(tensor_pd->dims()),
                     "Ensure ngraph tensor layout align with paddle tensor");
      auto ng_type = var_type_map_.at(vi);
      if (ng_type == ngraph::element::f32) {
        auto pd_arr = tensor_pd->mutable_data<float>(place);
        ti = backend_->create_tensor(ng_type, sp, pd_arr);
      } else if (ng_type == ngraph::element::i32) {
        auto pd_arr = tensor_pd->mutable_data<int>(place);
        ti = backend_->create_tensor(ng_type, sp, pd_arr);
      } else if (ng_type == ngraph::element::i64) {
        auto pd_arr = tensor_pd->mutable_data<int64_t>(place);
        ti = backend_->create_tensor(ng_type, sp, pd_arr);
      } else if (ng_type == ngraph::element::f64) {
        auto pd_arr = tensor_pd->mutable_data<double>(place);
        ti = backend_->create_tensor(ng_type, sp, pd_arr);
      } else if (ng_type == ngraph::element::boolean) {
        auto pd_arr = tensor_pd->mutable_data<bool>(place);
        ti = backend_->create_tensor(ng_type, sp, pd_arr);
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
    auto vo = var_out_[i];
    auto* var = scope.FindVar(vo);
    std::shared_ptr<ngraph::runtime::Tensor> to;
    if (var && var->IsType<framework::LoDTensor>()) {
      auto* tensor_pd = GetMutableLoDTensorOrSelectedRowsValueFromVar(var);
      auto dd = tensor_pd->dims();
      ngraph::Shape sp = Ddim2Shape(dd);
      auto ng_type = var_type_map_.at(vo);
      if (ng_type == ngraph::element::f32) {
        auto pd_arr = tensor_pd->mutable_data<float>(place);
        to = backend_->create_tensor(ng_type, sp, pd_arr);
      } else if (ng_type == ngraph::element::i64) {
        auto pd_arr = tensor_pd->mutable_data<int64_t>(place);
        to = backend_->create_tensor(ng_type, sp, pd_arr);
      } else if (ng_type == ngraph::element::f64) {
        auto pd_arr = tensor_pd->mutable_data<double>(place);
        to = backend_->create_tensor(ng_type, sp, pd_arr);
      } else if (ng_type == ngraph::element::boolean) {
        auto pd_arr = tensor_pd->mutable_data<bool>(place);
        to = backend_->create_tensor(ng_type, sp, pd_arr);
      } else {
        PADDLE_THROW("Data type not handled in for var %s", vo);
      }
      t_out.push_back(to);
    } else {
      PADDLE_THROW("Cannot find var or tensor with var name %s", vo);
    }
  }

  backend_->call(backend_->compile(ngraph_function_), t_out, t_in);
}  // NgraphEngine::RunImpl
}  // namespace operators
}  // namespace paddle
