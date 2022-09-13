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

  virtual void SetGraph(Scope* scope,
                        const ProgramDesc& prog,
                        const std::vector<std::string>& feed_names,
                        const std::vector<std::string>& fetch_names,
                        bool add_fetch_op) = 0;

  virtual paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names) = 0;
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

  void SetGraph(Scope* scope,
                const ProgramDesc& prog,
                const std::vector<std::string>& feed_names,
                const std::vector<std::string>& fetch_names,
                bool add_fetch_op) override {
    // 1. program key
    std::ostringstream oss;
    oss << "graph:" << &prog << ",";
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
      const std::vector<std::string>& feed_names) override {
    // phi::DeviceManager::SetDevice(place_);

    // for (size_t i = 0; i < feed_names.size(); ++i) {
    //   auto* feed_var = local_scope_->FindVar(feed_names[i]);
    //   PADDLE_ENFORCE_NOT_NULL(
    //       feed_var,
    //       platform::errors::NotFound("Variable %s should not be nullptr.",
    //                                  feed_names[i]));

    //   auto feed_tensor = feed_var->GetMutable<framework::LoDTensor>();
    //   feed_tensor->ShareDataWith(feed_tensors[i]);
    //   feed_tensor->set_lod(feed_tensors[i].lod());
    // }

    // phi::DeviceManager::GraphEnginePrepareGraph(
    //     place_,
    //     phi::stream::Stream(place_, nullptr),
    //     reinterpret_cast<void*>(copy_program_.get()),
    //     nullptr,
    //     nullptr,
    //     0);

    // feed
    std::vector<char*> feed_tensor_name;
    std::vector<void*> feed_tensor_data;

    auto* feed_var = local_scope_->FindVar("feed");
    if (feed_var) {
      auto* feed_list = feed_var->GetMutable<framework::FeedList>();
      for (size_t i = 0; i < feed_list->size(); ++i) {
        feed_tensor_name.push_back(const_cast<char*>(feed_names[i].c_str()));
        feed_tensor_data.push_back(paddle::get<0>(feed_list->at(i)).data());

        auto var_name = feed_names[i];
        auto var_dims = paddle::get<0>(feed_list->at(i)).dims();
        auto var = copy_program_->MutableBlock(0)->Var(feed_names[i]);
        var->SetShape(phi::vectorize<int64_t>(var_dims));
        copy_program_->MutableBlock(0)->Flush();
      }
    }

    // fetch
    std::vector<std::string> fetch_tensor_name_string;
    std::vector<char*> fetch_tensor_name;
    std::vector<void*> fetch_tensor_data;

    // infershape
    auto& block = *copy_program_->MutableBlock(0);

    for (auto& op_desc : block.AllOps()) {
      auto op_type = op_desc->Type();

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

      if (op->Type() == "fetch_v2") {
        auto in_var_name = op->Input("X");
        auto* in_var = local_scope_->FindVar(in_var_name);
        PADDLE_ENFORCE_NOT_NULL(
            in_var,
            platform::errors::NotFound("Input varibale(%s) cannot be found "
                                       "in scope for operator 'FetchV2'.",
                                       in_var_name));
        auto* out_var = local_scope_->FindVar(interpreter::kFetchVarName);
        auto* fetch_list = out_var->GetMutable<framework::FetchList>();
        auto col = op->Attr<int>("col");
        if (static_cast<size_t>(col) >= fetch_list->size()) {
          fetch_list->resize(col + 1);
        }

        if (in_var->IsType<framework::LoDTensor>()) {
          // auto& src_item = in_var->Get<framework::LoDTensor>();
          auto& fetch_item =
              paddle::get<0>(fetch_list->at(static_cast<size_t>(col)));
          fetch_item.Resize(
              phi::make_ddim(block.FindVar(in_var_name)->GetShape()));
          // fetch_item.set_lod(block.FindVar(in_var_name)->GetLoDLevel());
          auto fetch_data = fetch_item.mutable_data(
              paddle::CPUPlace(),
              TransToPhiDataType(block.FindVar(in_var_name)->GetDataType()));

          fetch_tensor_name_string.push_back(in_var_name);
          fetch_tensor_name.push_back(
              const_cast<char*>(fetch_tensor_name_string.back().c_str()));
          fetch_tensor_data.push_back(fetch_data);
        } else {
          PADDLE_THROW(platform::errors::Unavailable(
              "Unsupported OperatorBase %s", op->Type()));
        }
      } else if (dynamic_cast<framework::OperatorWithKernel*>(op) == nullptr &&
                 !cache_hit) {
        if (op->Type() == "feed") {
          OP_INOUT_CHECK(op->HasInputs("X"), "Input", "X", "Feed");
          OP_INOUT_CHECK(op->HasOutputs("Out"), "Output", "Out", "Feed");

          auto feed_var_name = op->Input("X");
          auto* feed_var = local_scope_->FindVar(feed_var_name);
          PADDLE_ENFORCE_NOT_NULL(
              feed_var,
              platform::errors::NotFound("Input varibale(%s) cannot be found "
                                         "in scope for operator 'Feed'.",
                                         feed_var_name));
          auto out_name = op->Output("Out");
          auto* out_var = local_scope_->FindVar(out_name);
          PADDLE_ENFORCE_NOT_NULL(
              out_var,
              platform::errors::NotFound("Output variable(%s) cannot be found "
                                         "in scope for operator 'Feed'",
                                         out_name));

          auto col = op->Attr<int>("col");
          PADDLE_ENFORCE_GE(
              col,
              0,
              platform::errors::InvalidArgument(
                  "Expected the column index (the attribute 'col' of "
                  "operator 'Feed') of current feeding variable to be "
                  "no less than 0. But received column index = %d.",
                  col));

          LOG(INFO) << "Feed variable " << feed_var_name << "'s " << col
                    << " column to variable " << out_name;

          auto& feed_list = feed_var->Get<framework::FeedList>();
          PADDLE_ENFORCE_LT(static_cast<size_t>(col),
                            feed_list.size(),
                            platform::errors::InvalidArgument(
                                "The column index of current feeding variable "
                                "is expected to be "
                                "less than the length of feeding list. But "
                                "received column index = "
                                "%d, the length of feeding list = %d",
                                col,
                                feed_list.size()));

          auto& feed_item =
              paddle::get<0>(feed_list.at(static_cast<size_t>(col)));
          auto out_tensor = out_var->GetMutable<framework::LoDTensor>();
          out_tensor->Resize(feed_item.dims());
          out_tensor->set_lod(feed_item.lod());
        } else {
          std::cerr << "unsupported operatorbase\n";
          exit(-1);
        }
      } else if (!cache_hit) {
        op_desc->InferShape(block);
      }
    }
    if (!cache_hit) {
      copy_program_->Flush();
    }

    // run graph
    phi::DeviceManager::GraphEngineExecuteGraph(
        place_,
        phi::stream::Stream(place_, nullptr),
        reinterpret_cast<void*>(copy_program_.get()),
        feed_tensor_name.data(),
        feed_tensor_data.data(),
        feed_tensor_data.size(),
        fetch_tensor_name.data(),
        fetch_tensor_data.data(),
        fetch_tensor_data.size());

    // return Fetch Tensors
    auto* fetch_var = local_scope_->FindVar(interpreter::kFetchVarName);
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
