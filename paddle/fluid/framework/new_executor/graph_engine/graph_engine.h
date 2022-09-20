// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/new_executor/interpretercore_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/profiler.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/ops_extra_info.h"
#include "paddle/phi/backends/device_manager.h"

namespace paddle {
namespace framework {
class GraphEngine {
 public:
  GraphEngine() = default;

  virtual ~GraphEngine() {}

  virtual void SetGraph(const ProgramDesc& prog,
                        const std::vector<std::string>& feed_names,
                        const std::vector<std::string>& fetch_names,
                        bool add_fetch_op) = 0;

  virtual paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<std::string>& fetch_names) = 0;
};

#ifdef PADDLE_WITH_CUSTOM_DEVICE

class CustomGraphEngine final : public GraphEngine {
 public:
  CustomGraphEngine(Scope* scope,
                    const ProgramDesc& prog,
                    const platform::Place& place)
      : place_(place) {
    // phi::DeviceManager::SetDevice(place_);

    VLOG(10) << "GE Initialize";

    phi::DeviceManager::GraphEngineInitialize(
        place_, phi::stream::Stream(place_, nullptr));

    var_scope_ = std::make_shared<VariableScope>(scope);

    auto local_scope = &var_scope_->GetMutableScope()->NewScope();
    local_scope_ = local_scope;

    var_scope_->SetLocalScope(local_scope_);
  }

  ~CustomGraphEngine() override {
    VLOG(10) << "GE Finalize";

    phi::DeviceManager::GraphEngineFinalize(
        place_, phi::stream::Stream(place_, nullptr));
  }

  void SetGraph(const ProgramDesc& prog,
                const std::vector<std::string>& feed_names,
                const std::vector<std::string>& fetch_names,
                bool add_fetch_op) override {
    std::ostringstream oss;
    oss << "program:" << &prog << ",";
    oss << "fetch:";
    for (auto& fetchname : fetch_names) {
      oss << fetchname << ",";
    }

    auto* feed_var = local_scope_->FindVar("feed");
    if (feed_var) {
      auto* feed_list = feed_var->GetMutable<framework::FeedList>();
      for (size_t i = 0; i < feed_list->size(); ++i) {
        auto var_name = feed_names[i];
        auto var_dims = paddle::get<0>(feed_list->at(i)).dims();
        oss << var_name << ":" << var_dims << ",";
      }
    }

    auto cached_program_key = oss.str();
    VLOG(10) << "get cached_program key: " << cached_program_key;

    if (cached_program_.find(cached_program_key) != cached_program_.end()) {
      VLOG(10) << "graph cache hit";
      copy_program_ = cached_program_[cached_program_key];
      cache_hit = true;
    } else {
      VLOG(10) << "graph cache miss";
      auto new_prog = std::make_shared<framework::ProgramDesc>(prog);

      if (add_fetch_op) {
        auto* block = new_prog->MutableBlock(0);
        interpreter::add_fetch(fetch_names, block);
      }

      copy_program_ = new_prog;
      cached_program_[cached_program_key] = copy_program_;
      cache_hit = false;

      paddle::framework::interpreter::build_variable_scope(
          copy_program_->Block(0), var_scope_.get(), true);

      for (auto& feed_name : feed_names) {
        var_scope_->SetVarSikpInplace(feed_name, true);
      }
    }
  }

  paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<std::string>& fetch_names) override {
    // phi::DeviceManager::SetDevice(place_);

    auto& block = *copy_program_->MutableBlock(0);

    // feed
    std::vector<char*> feed_tensor_name;
    std::vector<void*> feed_tensor_data;

    auto* feed_var = local_scope_->FindVar("feed");
    if (feed_var) {
      auto* feed_list = feed_var->GetMutable<framework::FeedList>();
      for (size_t i = 0; i < feed_list->size(); ++i) {
        auto& out_name = feed_names[i];
        if (!cache_hit) {
          auto* out_var = local_scope_->FindVar(out_name);
          auto& feed_item =
              paddle::get<0>(feed_list->at(static_cast<size_t>(i)));
          auto out_tensor = out_var->GetMutable<framework::LoDTensor>();
          out_tensor->Resize(feed_item.dims());
          out_tensor->set_lod(feed_item.lod());
          auto var = copy_program_->MutableBlock(0)->Var(out_name);
          var->SetShape(phi::vectorize<int64_t>(feed_item.dims()));
          // set_lo
        }
        feed_tensor_name.push_back(const_cast<char*>(out_name.c_str()));
        feed_tensor_data.push_back(paddle::get<0>(feed_list->at(i)).data());
      }
    }

    // infershape
    if (!cache_hit) {
      for (auto& op_desc : block.AllOps()) {
        auto op_type = op_desc->Type();
        if (op_type == "feed" || op_type == "fetch_v2") {
          continue;
        }

        VLOG(10) << "graph infershape for " << op_type;

        auto& info = OpInfoMap::Instance().Get(op_type);

        const VariableNameMap& inputs_names = op_desc->Inputs();
        const VariableNameMap& outputs_names = op_desc->Outputs();

        AttributeMap op_attr_map = op_desc->GetAttrMap();
        AttributeMap op_runtime_attr_map = op_desc->GetRuntimeAttrMap();

        if (info.Checker() != nullptr) {
          info.Checker()->Check(&op_attr_map);
        }

        const auto& extra_attr_checkers =
            operators::ExtraInfoUtils::Instance().GetExtraAttrsChecker(op_type);
        for (const auto& checker : extra_attr_checkers) {
          checker(&op_runtime_attr_map, false);
        }

        auto op =
            info.Creator()(op_type, inputs_names, outputs_names, op_attr_map);
        op->SetRuntimeAttributeMap(op_runtime_attr_map);

        if (dynamic_cast<framework::OperatorWithKernel*>(op) == nullptr) {
          PADDLE_THROW(platform::errors::Unavailable(
              "Unsupported OperatorBase %s", op->Type()));
        } else {
          op_desc->InferShape(block);
        }
      }
      copy_program_->Flush();
    }

    // fetch
    std::vector<char*> fetch_tensor_name;
    std::vector<void*> fetch_tensor_data;
    auto* fetch_var = local_scope_->FindVar(interpreter::kFetchVarName);
    if (fetch_var) {
      auto* fetch_list = fetch_var->GetMutable<framework::FetchList>();
      fetch_list->resize(fetch_names.size());
      for (size_t i = 0; i < fetch_list->size(); ++i) {
        auto* in_var = local_scope_->FindVar(fetch_names[i]);

        if (in_var->IsType<framework::LoDTensor>()) {
          auto& fetch_item =
              paddle::get<0>(fetch_list->at(static_cast<size_t>(i)));
          fetch_item.Resize(
              phi::make_ddim(block.FindVar(fetch_names[i])->GetShape()));
          // set_lod
          auto fetch_item_data = fetch_item.mutable_data(
              paddle::CPUPlace(),
              TransToPhiDataType(block.FindVar(fetch_names[i])->GetDataType()));

          fetch_tensor_name.push_back(
              const_cast<char*>(fetch_names[i].c_str()));
          fetch_tensor_data.push_back(fetch_item_data);
        } else {
          PADDLE_THROW(platform::errors::Unavailable(
              "Unsupported Variable Type %d", in_var->Type()));
        }
      }
    }

    // run graph
    phi::DeviceManager::GraphEngineExecuteGraph(
        place_,
        phi::stream::Stream(place_, nullptr),
        reinterpret_cast<void*>(local_scope_),
        reinterpret_cast<void*>(copy_program_.get()),
        feed_tensor_name.data(),
        feed_tensor_data.data(),
        feed_tensor_data.size(),
        fetch_tensor_name.data(),
        fetch_tensor_data.data(),
        fetch_tensor_data.size());

    // return Fetch Tensors
    if (fetch_var) {
      return std::move(*fetch_var->GetMutable<framework::FetchList>());
    } else {
      return {};
    }
  }

 private:
  platform::Place place_;

  std::shared_ptr<VariableScope> var_scope_{nullptr};

  Scope* local_scope_{nullptr};  // not owned

  std::shared_ptr<ProgramDesc> copy_program_{nullptr};

  std::unordered_map<std::string, std::shared_ptr<ProgramDesc>>
      cached_program_{};

  bool cache_hit{false};
};

#endif

}  // namespace framework
}  // namespace paddle
