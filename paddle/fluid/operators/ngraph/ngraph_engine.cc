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
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/ngraph/ngraph_bridge.h"
#include "paddle/fluid/operators/ngraph/ngraph_engine.h"

namespace paddle {
namespace operators {

static ngraph::Shape Ddim2Shape(const framework::DDim& dims) {
  ngraph::Shape sp;
  for (int i = 0; i < dims.size(); ++i) {
    sp.emplace_back(dims[i]);
  }
  return sp;
}

static framework::DDim Shape2Ddim(const ngraph::Shape& shape) {
  std::vector<int64_t> dims;
  for (size_t i = 0; i < shape.size(); ++i) {
    int64_t k = shape[i];
    dims.emplace_back(k);
  }
  return framework::make_ddim(dims);
}

static std::map<framework::proto::VarType::Type, ngraph::element::Type>
    pd2ng_type_map = {
        {framework::proto::VarType::FP32, ngraph::element::f32},
        {framework::proto::VarType::FP64, ngraph::element::f64},
        {framework::proto::VarType::INT32, ngraph::element::i32},
        {framework::proto::VarType::INT64, ngraph::element::i64},
        {framework::proto::VarType::UINT8, ngraph::element::u8},
        {framework::proto::VarType::BOOL, ngraph::element::boolean}};

static std::map<ngraph::element::Type, framework::proto::VarType::Type>
    ng2pd_type_map = {
        {ngraph::element::f32, framework::proto::VarType::FP32},
        {ngraph::element::f64, framework::proto::VarType::FP64},
        {ngraph::element::i32, framework::proto::VarType::INT32},
        {ngraph::element::i64, framework::proto::VarType::INT64},
        {ngraph::element::u8, framework::proto::VarType::UINT8},
        {ngraph::element::boolean, framework::proto::VarType::BOOL}};

std::vector<std::string> NgraphEngine::feed_vars = {};
std::vector<std::string> NgraphEngine::fetch_vars = {};
framework::Variable* NgraphEngine::pre_var_ptr = nullptr;
const framework::BlockDesc* NgraphEngine::p_bdesc = nullptr;
bool NgraphEngine::is_training = false;

std::shared_ptr<ngraph::runtime::Backend> NgraphEngine::backend_ =
    ngraph::runtime::Backend::create("CPU");

static std::vector<std::vector<int>> NgraphOpIntervals(
    std::vector<std::unique_ptr<framework::OperatorBase>>* ops) {
  NgraphEngine::feed_vars.clear();
  NgraphEngine::fetch_vars.clear();
  std::vector<std::vector<int>> intervals;

  int size = ops->size();
  int left = 0, feed_idx = -1;
  while (left < size && ops->at(left)->Type() != framework::kFeedOpType &&
         ops->at(left)->Type() != "read" &&
         ops->at(left)->Type() != framework::kFetchOpType) {
    ++left;
  }

  if (left < size) {
    auto op_type = ops->at(left)->Type();
    if (op_type == framework::kFeedOpType || op_type == "read") {
      feed_idx = left;
    }
  }

  while (left < size && (ops->at(left)->Type() == framework::kFeedOpType ||
                         ops->at(left)->Type() == "read")) {
    for (auto& var_name_item : ops->at(left)->Outputs()) {
      for (auto& var_name : var_name_item.second) {
        NgraphEngine::feed_vars.emplace_back(var_name);
      }
    }
    ++left;
  }

  int right = left;
  while (right < size && ops->at(right)->Type() != framework::kFetchOpType) {
    ++right;
  }

  int index = right;
  while (index < size && ops->at(index)->Type() == framework::kFetchOpType) {
    for (auto& var_name_item : ops->at(index)->Inputs()) {
      for (auto& var_name : var_name_item.second) {
        NgraphEngine::fetch_vars.emplace_back(var_name);
      }
    }
    ++index;
  }

  if (left == size || ops->at(left)->Type() == framework::kFetchOpType) {
    left = 0;
  }

  // (left, right - 1) represents indices between feed and fetch
  int pivot = left;
  while (pivot < right) {
    auto op_type = ops->at(pivot)->Type();
    if (!NgraphBridge::isSupported(ops->at(pivot))) {
      ++pivot;
    } else {
      int start = pivot, end = start;
      while (pivot < right && (NgraphBridge::isSupported(ops->at(pivot)))) {
        ++pivot;
        ++end;
      }
      std::vector<int> interval = {start, end};
      if (feed_idx != -1 && start > feed_idx) {
        intervals.emplace_back(interval);
      }
    }
  }  // end while
  return intervals;
}

static void SubstituteNgraphOp(
    std::vector<std::unique_ptr<framework::OperatorBase>>* ops,
    std::string engine_key, std::string block_str, std::vector<int> interval) {
  framework::OpDesc ng_op_desc(nullptr);
  ng_op_desc.SetType("ngraph_engine");
  ng_op_desc.SetAttr("interval", interval);
  ng_op_desc.SetAttr("engine_key", engine_key);
  ng_op_desc.SetAttr("graph", block_str);
  ng_op_desc.SetInput("Xs", std::vector<std::string>(0));
  ng_op_desc.SetOutput("Ys", std::vector<std::string>(0));

  ops->erase(ops->begin() + interval[0], ops->begin() + interval[1]);
  ops->insert(ops->begin() + interval[0],
              framework::OpRegistry::CreateOp(ng_op_desc));
}

std::string SerializedBlock(const std::vector<framework::OpDesc*>& op_descs) {
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);

  for (auto* op_desc : op_descs) {
    auto* op = block_desc.AppendOp();
    *op->Proto() = *op_desc->Proto();
  }
  return block_desc.Proto()->SerializeAsString();
}

std::string GenerateEngineKey(const framework::BlockDesc& bdesc) {
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);

  for (auto& op_desc : bdesc.AllOps()) {
    auto* op = block_desc.AppendOp();
    *op->Proto() = *op_desc->Proto();
  }
  auto engine_key = std::to_string(
      std::hash<std::string>()(block_desc.Proto()->SerializeAsString()));
  return engine_key;
}

std::string GenerateEngineKey(const std::vector<std::string>& engine_inputs,
                              const std::vector<std::string>& engine_outputs,
                              int size) {
  std::string engine_hash_key = "";
  for (auto name : engine_inputs) {
    engine_hash_key += name;
  }
  for (auto name : engine_outputs) {
    engine_hash_key += name;
  }
  engine_hash_key += std::to_string(size);
  auto engine_key = std::to_string(std::hash<std::string>()(engine_hash_key));
  return engine_key;
}

void NgraphEngine::FuseNgraphOps(
    const framework::BlockDesc& block_desc,
    std::vector<std::unique_ptr<framework::OperatorBase>>* ops) {
  NgraphEngine::p_bdesc = &block_desc;
  auto intervals = NgraphOpIntervals(ops);
  std::string engine_key =
      GenerateEngineKey(feed_vars, fetch_vars, ops->size());
  for (auto it = intervals.rbegin(); it != intervals.rend(); ++it) {
    SubstituteNgraphOp(ops, engine_key, "", *it);
  }
}

NgraphEngine::NgraphEngine(const framework::Scope& scope,
                           const platform::Place& place,
                           const framework::ExecutionContext& ctx)
    : scope_(scope), place_(place) {
  var_in_node_map_ = std::make_shared<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>();

  var_node_map_ = std::make_shared<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>();

  GetNgFunction(ctx);
}

void NgraphEngine::Prepare(const framework::ExecutionContext& ctx) {
  auto interval = ctx.Attr<std::vector<int>>("interval");
  std::string serialized_graph = ctx.Attr<std::string>("graph");

  auto input_vars = ctx.Inputs("Xs");
  if (!input_vars.empty()) {
    feed_vars = input_vars;
    var_in_ = input_vars;
  }
  auto output_vars = ctx.Outputs("Ys");
  if (!output_vars.empty()) {
    var_out_ = output_vars;
  }

  framework::proto::BlockDesc block_proto;
  if (!serialized_graph.empty()) block_proto.ParseFromString(serialized_graph);
  framework::BlockDesc block_desc(nullptr, &block_proto);
  if (!serialized_graph.empty()) {
    NgraphEngine::p_bdesc = &block_desc;
  }

  for (auto& var : p_bdesc->AllVars()) {
    if (!(var->GetType() == framework::proto::VarType::SELECTED_ROWS ||
          var->GetType() == framework::proto::VarType::LOD_TENSOR ||
          var->GetType() == framework::proto::VarType::LOD_TENSOR_ARRAY)) {
      continue;
    }

    auto var_name = var->Name();
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }

    if (var_name != framework::kFeedOpType &&
        var_name != framework::kFetchOpType) {
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

  std::vector<paddle::framework::OpDesc*> ops_desc;
  for (auto op_desc : p_bdesc->AllOps()) {
    ops_desc.emplace_back(op_desc);
    if (op_desc->Type().find("_grad") != std::string::npos) {
      is_training = true;
      this->is_test_ = false;
    }
  }

  int idx = interval[0];
  while (idx < interval[1]) {
    this->fused_ops_.emplace_back(
        framework::OpRegistry::CreateOp(*(ops_desc[idx])));
    ++idx;
  }
  while (idx < static_cast<int>(ops_desc.size()) &&
         ops_desc.at(idx)->Type() != framework::kFetchOpType) {
    auto op_desc = ops_desc.at(idx);
    for (auto& var_name_item : op_desc->Inputs()) {
      for (auto& var_name : var_name_item.second) {
        this->post_op_inputs_.insert(var_name);
      }
    }
    ++idx;
  }

  if (var_in_.empty() && var_out_.empty()) {
    BuildNgIO(ops_desc, interval);
  }
  for (size_t i = 0; i < var_in_.size(); ++i) {
    auto var_name = var_in_[i];
    if (persistables_.find(var_name) == persistables_.end()) {
      var_in_updates_.emplace_back(i);
    }
  }
}

void NgraphEngine::BuildNgIO(const std::vector<framework::OpDesc*>& ops_desc,
                             const std::vector<int>& interval) {
  std::unordered_set<std::string> inputs;
  std::unordered_set<std::string> outputs;
  for (int i = interval[0]; i < interval[1]; ++i) {
    auto op = ops_desc[i];
    for (auto& var_name_item : op->Inputs()) {
      for (auto& var_name : var_name_item.second) {
        inputs.insert(var_name);
        const bool is_output = outputs.find(var_name) != outputs.end();
        if (!is_output &&
            std::find(var_in_.begin(), var_in_.end(), var_name) ==
                var_in_.end() &&
            scope_.FindVar(var_name)) {
          // fill var_in here to keep lhs and rhs order
          this->var_in_.emplace_back(var_name);
        }
      }
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
  for (int i = interval[0]; i < interval[1]; ++i) {
    auto op = ops_desc[i];
    for (auto& var_name_item : op->Outputs()) {
      PADDLE_ENFORCE_LE(var_name_item.second.size(), 1,
                        "op %s has more than 1 output - Not handling yet",
                        op->Type());
      for (auto& var_name : var_name_item.second) {
        if (this->is_test_) {
          if (post_op_inputs_.find(var_name) != post_op_inputs_.end() ||
              find(fetch_vars.begin(), fetch_vars.end(), var_name) !=
                  fetch_vars.end()) {
            this->var_out_.emplace_back(var_name);
          }
        } else {
          if (find(fetch_vars.begin(), fetch_vars.end(), var_name) !=
                  fetch_vars.end() ||
              post_op_inputs_.find(var_name) != post_op_inputs_.end() ||
              persistables_.find(var_name) != persistables_.end()) {
            this->var_out_.emplace_back(var_name);
          }
        }
      }
    }
  }
  // remove output duplicates
  std::unordered_set<std::string> var_out_set;
  for (int i = static_cast<int>(var_out_.size()) - 1; i >= 0; --i) {
    std::string var_name = var_out_.at(i);
    if (var_out_set.count(var_name)) {
      var_out_.erase(var_out_.begin() + i);
    }
    var_out_set.insert(var_name);
  }
}

void NgraphEngine::GetNgInputShape() {
  for (auto& var_name : var_in_) {
    auto* var = scope_.FindVar(var_name);
    if (var && var->IsType<framework::LoDTensor>()) {
      auto* tensor_pd = GetLoDTensorOrSelectedRowsValueFromVar(*var);
      auto sp = Ddim2Shape(tensor_pd->dims());
      auto ng_type = var_type_map_[var_name];
      auto prm = std::make_shared<ngraph::op::Parameter>(ng_type, sp, true);
      (*var_node_map_)[var_name] = prm;
      (*var_in_node_map_)[var_name] = prm;
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
            auto ng_type = var_type_map_[var_name];
            auto prm = std::make_shared<ngraph::op::Parameter>(ng_type,
                                                               ng_shape, true);
            (*var_node_map_)[var_name] = prm;
          }
        }
      }
    }
  }
  NgraphBridge ngb(var_node_map_);
  for (auto& op : fused_ops_) {
    ngb.BuildNgNode(op);
  }
}

std::shared_ptr<ngraph::Function> NgraphEngine::BuildNgFunction(
    const framework::ExecutionContext& ctx) {
  Prepare(ctx);
  GetNgInputShape();
  BuildNgNodes();
  ngraph::NodeVector func_outputs;
  ngraph::ParameterVector func_inputs;

  for (auto& vo : var_out_) {
    func_outputs.emplace_back(var_node_map_->at(vo));
  }

  for (auto& vi : var_in_) {
    std::shared_ptr<ngraph::op::Parameter> prm =
        std::dynamic_pointer_cast<ngraph::op::Parameter>(
            var_in_node_map_->at(vi));
    func_inputs.emplace_back(prm);
  }

  return std::make_shared<ngraph::Function>(func_outputs, func_inputs);
}

void NgraphEngine::ClearNgCache() {
  auto& engine_cache = main_engine_cache::fetch();
  auto& t_in_cache_ = main_t_in_cache::fetch();

  auto it = engine_cache.begin();
  while (it != engine_cache.end()) {
    auto ng_engine = it->second;
    backend_->remove_compiled_function(ng_engine.ngraph_handle);
    ++it;
  }
  engine_cache.clear();
  auto it_tensor = t_in_cache_.begin();
  while (it_tensor != t_in_cache_.end()) {
    auto t_vec = it_tensor->second;
    for (auto t_in : t_vec) {
      t_in.reset();
    }
    ++it_tensor;
  }
  t_in_cache_.clear();
}

void NgraphEngine::GetNgFunction(const framework::ExecutionContext& ctx) {
  auto interval = ctx.Attr<std::vector<int>>("interval");
  std::string engine_key = ctx.Attr<std::string>("engine_key");

  // set to flase, to debug cache or recompile everytime.
  bool use_cache = true;
  if (!use_cache) ClearNgCache();

  this->func_cache_key_ = "";
  for (int i = 0; i < static_cast<int>(feed_vars.size()); ++i) {
    auto* var = scope_.FindVar(feed_vars[i]);
    if (var && var->IsType<framework::LoDTensor>()) {
      auto* tensor_pd = GetLoDTensorOrSelectedRowsValueFromVar(*var);
      auto dims = tensor_pd->dims();
      for (int j = 0; j < dims.size(); ++j) {
        func_cache_key_ += std::to_string(dims[j]);
      }
    }
  }
  func_cache_key_ += std::to_string(interval[0]) + "_" +
                     std::to_string(interval[1]) + engine_key;
  func_cache_key_ = std::to_string(std::hash<std::string>()(func_cache_key_));

  auto& engine_cache = main_engine_cache::fetch();

  if (engine_cache.find(func_cache_key_) != engine_cache.end()) {
    if (engine_cache[func_cache_key_].persistables.size() == 0) {
      ClearNgCache();
    } else {
      auto var_name = engine_cache[func_cache_key_].persistables.begin();
      framework::Variable* var = scope_.FindVar(*var_name);
      if (var != pre_var_ptr) {
        ClearNgCache();
      }
      pre_var_ptr = var;
    }
  }

  if (engine_cache.find(func_cache_key_) == engine_cache.end()) {
    if (engine_cache.size() > 5) ClearNgCache();
    auto func = BuildNgFunction(ctx);
    // Due to optimization backend may produce results in other layouts,
    // make sure we get default layout for results.
    for (auto& r : func->get_results()) {
      r->set_needs_default_layout(true);
    }
    engine_cache[func_cache_key_].ngraph_handle = backend_->compile(func);
    engine_cache[func_cache_key_].persistables = this->persistables_;
    engine_cache[func_cache_key_].var_in_updates = this->var_in_updates_;
    engine_cache[func_cache_key_].var_in = this->var_in_;
    engine_cache[func_cache_key_].var_out = this->var_out_;
    engine_cache[func_cache_key_].is_test = this->is_test_;
  }
}

void NgraphEngine::Run(const framework::Scope& scope,
                       const platform::Place& place) const {
  std::shared_ptr<ngraph::runtime::Executable> ng_handle;
  const std::set<std::string>* p_persistables;
  const std::vector<size_t>* p_var_in_updates;
  const std::vector<std::string>* p_var_in;
  const std::vector<std::string>* p_var_out;
  bool is_test;

  auto& engine_cache = main_engine_cache::fetch();
  auto& t_in_cache_ = main_t_in_cache::fetch();

  PADDLE_ENFORCE(engine_cache.find(func_cache_key_) != engine_cache.end(),
                 "Cannot find cached data to run ngraph function");
  ng_handle = engine_cache[func_cache_key_].ngraph_handle;
  p_persistables = &(engine_cache[func_cache_key_].persistables);
  p_var_in_updates = &(engine_cache[func_cache_key_].var_in_updates);
  p_var_in = &(engine_cache[func_cache_key_].var_in);
  p_var_out = &(engine_cache[func_cache_key_].var_out);
  is_test = engine_cache[func_cache_key_].is_test;

  std::vector<std::shared_ptr<ngraph::runtime::Tensor>>* p_t_in;
  std::vector<std::shared_ptr<ngraph::runtime::Tensor>> t_in = {};

  auto m_parameters = ng_handle->get_parameters();
  auto m_results = ng_handle->get_results();
  if (is_test && t_in_cache_.find(func_cache_key_) != t_in_cache_.end()) {
    p_t_in = &(t_in_cache_[func_cache_key_]);
    for (size_t i = 0; i < p_var_in_updates->size(); ++i) {
      int index = p_var_in_updates->at(i);
      auto vi = p_var_in->at(index);
      auto sp = m_parameters[index]->get_shape();
      auto ng_type = m_parameters[index]->get_element_type();
      std::shared_ptr<ngraph::runtime::Tensor> ti;
      auto* var = scope.FindVar(vi);
      if (var && var->IsType<framework::LoDTensor>()) {
        auto* tensor_pd = GetMutableLoDTensorOrSelectedRowsValueFromVar(var);
        void* pd_arr = tensor_pd->mutable_data(place, ng2pd_type_map[ng_type]);
        ti = backend_->create_tensor(ng_type, sp, pd_arr);
        (*p_t_in)[index] = ti;
      } else {
        PADDLE_THROW("Cannot find var or tensor with var name %s", vi);
      }
    }
  } else {
    if (is_test) {
      p_t_in = &(t_in_cache_[func_cache_key_]);
    } else {
      p_t_in = &t_in;
    }

    for (size_t i = 0; i < p_var_in->size(); ++i) {
      auto vi = p_var_in->at(i);
      auto sp = m_parameters[i]->get_shape();
      auto ng_type = m_parameters[i]->get_element_type();
      std::shared_ptr<ngraph::runtime::Tensor> ti;
      auto* var = scope.FindVar(vi);
      if (var && var->IsType<framework::LoDTensor>()) {
        auto* tensor_pd = GetMutableLoDTensorOrSelectedRowsValueFromVar(var);
        void* pd_arr = tensor_pd->mutable_data(place, ng2pd_type_map[ng_type]);
        PADDLE_ENFORCE(sp == Ddim2Shape(tensor_pd->dims()),
                       "Ensure ngraph tensor layout align with paddle tensor");
        ti = backend_->create_tensor(ng_type, sp, pd_arr);
      } else {
        PADDLE_THROW("Cannot find var or tensor with var name %s", vi);
      }
      bool is_persistable =
          (p_persistables->find(vi) != p_persistables->end()) ? true : false;
      if (!is_training && is_test && is_persistable) {
        ti->set_stale(false);
      }
      (*p_t_in).emplace_back(ti);
    }
  }

  for (auto& op : fused_ops_) {
    framework::RuntimeContext ctx(op->Inputs(), op->Outputs(), scope_);
    op->RuntimeInferShape(scope_, place_, ctx);
  }

  std::vector<std::shared_ptr<ngraph::runtime::Tensor>> t_out = {};
  for (size_t i = 0; i < p_var_out->size(); ++i) {
    auto vo = p_var_out->at(i);
    auto* var = scope.FindVar(vo);
    if (var && var->IsType<framework::LoDTensor>()) {
      auto sp = m_results[i]->get_shape();
      var->GetMutable<framework::LoDTensor>()->Resize(Shape2Ddim(sp));
      auto* tensor_pd = GetMutableLoDTensorOrSelectedRowsValueFromVar(var);
      auto ng_type = m_results[i]->get_element_type();
      void* pd_arr = tensor_pd->mutable_data(place, ng2pd_type_map[ng_type]);
      std::shared_ptr<ngraph::runtime::Tensor> to =
          backend_->create_tensor(ng_type, sp, pd_arr);
      t_out.emplace_back(to);
    } else {
      PADDLE_THROW("Cannot find var or tensor with var name %s", vo);
    }
  }

  ng_handle->call(t_out, *p_t_in);
}  // NgraphEngine::Run
}  // namespace operators
}  // namespace paddle
