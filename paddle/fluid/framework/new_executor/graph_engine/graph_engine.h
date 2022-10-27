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
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/profiler.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/ops_extra_info.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"
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
        interpreter::AddFetch(fetch_names, block);
      }

      copy_program_ = new_prog;
      cached_program_[cached_program_key] = copy_program_;
      cache_hit = false;

      paddle::framework::interpreter::BuildVariableScope(
          copy_program_->Block(0), var_scope_.get(), true);

      for (auto& feed_name : feed_names) {
        var_scope_->SetVarSikpInplace(feed_name, true);
      }
    }
  }

  paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<std::string>& fetch_names) override {
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
          auto out_tensor = out_var->GetMutable<phi::DenseTensor>();
          out_tensor->Resize(feed_item.dims());
          out_tensor->set_lod(feed_item.lod());
          auto var = copy_program_->MutableBlock(0)->Var(out_name);
          var->SetShape(phi::vectorize<int64_t>(feed_item.dims()));
          // set_lod
        }
        feed_tensor_name.push_back(const_cast<char*>(out_name.c_str()));
        feed_tensor_data.push_back(paddle::get<0>(feed_list->at(i)).data());
      }
    }

    if (!cache_hit) {
      auto var_scope = var_scope_.get();
      auto& local_scope = local_scope_;

      std::vector<std::shared_ptr<OperatorBase>> ops;

      // Step 1: create all ops for current block.
      for (auto& op : block.AllOps()) {
        auto op_type = op->Type();
        VLOG(8) << "CreateOp from : " << op_type;

        auto& info = OpInfoMap::Instance().Get(op_type);

        const VariableNameMap& inputs_names = op->Inputs();
        const VariableNameMap& outputs_names = op->Outputs();

        AttributeMap op_attr_map = op->GetAttrMap();
        AttributeMap op_runtime_attr_map = op->GetRuntimeAttrMap();

        if (info.Checker() != nullptr) {
          info.Checker()->Check(&op_attr_map);
        }

        const auto& extra_attr_checkers =
            operators::ExtraInfoUtils::Instance().GetExtraAttrsChecker(op_type);
        for (const auto& checker : extra_attr_checkers) {
          checker(&op_runtime_attr_map, false);
        }

        auto op_base =
            info.Creator()(op_type, inputs_names, outputs_names, op_attr_map);
        op_base->SetRuntimeAttributeMap(op_runtime_attr_map);

        ops.emplace_back(std::shared_ptr<OperatorBase>(op_base));
      }

      // auto unused_var_map = interpreter::GetUnusedVars(block, ops);

      for (size_t i = 0; i < ops.size(); ++i) {
        auto op = ops[i].get();
        const std::string& op_type = op->Type();
        if (op_type == "feed" || op_type == "fetch_v2") {
          continue;
        }

        VLOG(6) << "Build OpFuncNode from : " << op_type;

        // Hot fix for variables used in dataloader, like
        // 'lod_tensor_blocking_queue_0'. These variables may be created in
        // scope, and it is not existed as variable in program.
        const std::set<std::string> ops_with_var_not_in_program = {
            "create_py_reader"};
        const std::set<std::string> ops_with_var_not_in_scope = {
            "conditional_block",
            "conditional_block_grad",
            "recurrent_grad",
            "rnn_memory_helper",
            "rnn_memory_helper_grad",
            "while",
            "while_grad"};
        bool allow_var_not_in_program =
            ops_with_var_not_in_program.count(op_type);
        bool allow_var_not_in_scope = ops_with_var_not_in_scope.count(op_type);

        framework::VariableNameMap& input_name_map = op->Inputs();
        VariableValueMap ins_map;
        std::map<std::string, std::vector<int>> ins_name2id;
        std::tie(ins_map, ins_name2id) =
            interpreter::BuildVariableMap(input_name_map,
                                          var_scope,
                                          local_scope,
                                          allow_var_not_in_program,
                                          allow_var_not_in_scope);

        framework::VariableNameMap& output_name_map = op->Outputs();
        VariableValueMap outs_map;
        std::map<std::string, std::vector<int>> outs_name2id;
        std::tie(outs_map, outs_name2id) =
            interpreter::BuildVariableMap(output_name_map,
                                          var_scope,
                                          local_scope,
                                          /*allow_var_not_in_program=*/false,
                                          allow_var_not_in_scope);

        // step 1: build OpFuncNode
        OpFuncNode op_func_node;
        op_func_node.operator_base_ = ops[i];
        op_func_node.input_index = ins_name2id;
        op_func_node.output_index = outs_name2id;
        VLOG(4) << "Start run " << place_ << " "
                << op->DebugStringEx(local_scope);

        if (dynamic_cast<framework::OperatorWithKernel*>(op) == nullptr) {
          VLOG(4) << "HandleOperatorBase";
          PADDLE_THROW(
              platform::errors::Fatal("Not found phi kernel for %s", op_type));
          // op is not a operatorwithkernel, so direcly run OperatorBase::Run()
          // HandleOperatorBase(
          //     place_, var_scope, ops[i], &op_func_node, local_scope);
        } else {
          VLOG(4) << "OP is not null";
          auto op_with_kernel = const_cast<framework::OperatorWithKernel*>(
              static_cast<const framework::OperatorWithKernel*>(op));
          VLOG(4) << "get op_with_kernel";
          // construct RuntimeContext and analysis KernelType
          RuntimeContext runtime_context({}, {});
          runtime_context.inputs.swap(ins_map);
          runtime_context.outputs.swap(outs_map);
          VLOG(4) << "get RuntimeContext";

          Scope scope, *runtime_scope = &scope;

          auto& pool = platform::DeviceContextPool::Instance();
          auto* dev_ctx = pool.Get(place_);
          VLOG(4) << "get dev_ctx";
          auto exec_ctx = ExecutionContext(
              *op_with_kernel, *runtime_scope, *dev_ctx, runtime_context);
          VLOG(4) << "get exec_ctx";
          auto expected_kernel_key =
              op_with_kernel->GetExpectedKernelType(exec_ctx);
          VLOG(4) << "get expected_kernel_key";

          VLOG(4) << "expected_kernel_key : " << expected_kernel_key;

          // step 2. select op kernel
          auto run_phi_kernel = false;
          if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(
                  op_with_kernel->Type())) {
            auto phi_kernel_key = op_with_kernel->ChoosePhiKernel(exec_ctx);
            auto phi_kernel_name = op_with_kernel->PhiKernelSignature()->name;

            if (op_with_kernel->PhiKernel()->IsValid()) {
              run_phi_kernel = true;
            } else {
              run_phi_kernel = false;
            }
          }

          VLOG(4) << "if run phi kernel? : " << run_phi_kernel;
          if (!run_phi_kernel) {
            PADDLE_THROW(platform::errors::Fatal("Not found phi kernel for %s",
                                                 op_type));
            // op_with_kernel->ChooseKernel(exec_ctx);
            // op_func_node.kernel_func_ = *op_with_kernel->kernel_func();
          } else {
            op_func_node.phi_kernel_ = op_with_kernel->PhiKernel();
          }
          auto kernel_type = *(op_with_kernel->kernel_type());
          if (kernel_type.place_ != dev_ctx->GetPlace()) {
            dev_ctx = pool.Get(kernel_type.place_);
          }
          op_func_node.dev_ctx_ = dev_ctx;
          VLOG(3) << op_with_kernel->Type()
                  << " : finally selected kernel_key: " << kernel_type;

          InterpretercoreInferShapeContext infer_shape_ctx(*op,
                                                           runtime_context);
          op_with_kernel->Info().infer_shape_(&infer_shape_ctx);

          // step 5. run kernel
          if (run_phi_kernel) {
            phi::KernelContext phi_kernel_context;
            op_with_kernel->BuildPhiKernelContext(
                runtime_context, dev_ctx, &phi_kernel_context);
            (*op_func_node.phi_kernel_)(&phi_kernel_context);
          } else {
            // the place of exec_ctx maybe has changed.
            op_func_node.kernel_func_(ExecutionContext(
                *op_with_kernel, *runtime_scope, *dev_ctx, runtime_context));
          }
        }
      }
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

        if (in_var->IsType<phi::DenseTensor>()) {
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

  ~CustomGraphEngine() override {
    VLOG(10) << "GE Finalize";

    phi::DeviceManager::GraphEngineFinalize(
        place_, phi::stream::Stream(place_, nullptr));
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

class DeprecatedCustomGraphEngine final : public GraphEngine {
 public:
  DeprecatedCustomGraphEngine(Scope* scope,
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

  ~DeprecatedCustomGraphEngine() override {
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
        interpreter::AddFetch(fetch_names, block);
      }

      copy_program_ = new_prog;
      cached_program_[cached_program_key] = copy_program_;
      cache_hit = false;

      paddle::framework::interpreter::BuildVariableScope(
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
          auto out_tensor = out_var->GetMutable<phi::DenseTensor>();
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

        if (in_var->IsType<phi::DenseTensor>()) {
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
