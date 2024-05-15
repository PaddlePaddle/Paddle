/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/gcu/executor/single_op_executor.h"

#include <algorithm>
#include <chrono>  // NOLINT [build/c++11]
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gcu/profile/profile.h"
#include "paddle/fluid/platform/device/gcu/runtime/gcu_rt_interface.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_executable.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_stream.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_utils.h"
#include "paddle/fluid/platform/device/gcu/utils/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace platform {
namespace gcu {
using GcuCtxPtr = paddle::platform::gcu::runtime::GcuCtxPtr;
using GcuStreamPtr = paddle::platform::gcu::runtime::GcuStreamPtr;
using ExecutablePtr = paddle::platform::gcu::runtime::ExecutablePtr;
using Place = paddle::platform::Place;
using CustomPlace = paddle::platform::CustomPlace;
using DDim = paddle::framework::DDim;
using TensorPtr = std::shared_ptr<phi::DenseTensor>;
using paddle::framework::ExecutionContext;
using paddle::framework::Scope;
using phi::DenseTensor;
using LoDTensor = phi::DenseTensor;

// using namespace std;          // NOLINT
// using namespace std::chrono::system_clock;  // NOLINT

namespace {
const char* const kProfilerValue = std::getenv(kProfiler);
const size_t kSingleOpGraphId = 0;

std::chrono::time_point<std::chrono::system_clock> Now() {
  return std::chrono::system_clock::now();
}

void InitTensor(TensorPtr& tensor,   // NOLINT
                const Place& place,  // NOLINT
                const PaddleVarDesc& var_desc) {
  PADDLE_ENFORCE_NOT_NULL(tensor);
  tensor->Resize(phi::make_ddim(var_desc.shapes));
  tensor->mutable_data(place,
                       framework::TransToPhiDataType(var_desc.data_type));
}

TensorPtr CreateTensor(const Place& place, const PaddleVarDesc& var_desc) {
  auto tensor_ptr = std::make_shared<Tensor>();
  PADDLE_ENFORCE_NOT_NULL(tensor_ptr);
  InitTensor(tensor_ptr, place, var_desc);
  return tensor_ptr;
}

TensorPtr CreateTensorSharedOrFrom(const Tensor* tensor,
                                   const Place& place,
                                   TensorPtr& dst_ternsor) {  // NOLINT
  if (dst_ternsor == nullptr) {
    dst_ternsor = std::make_shared<Tensor>();
    PADDLE_ENFORCE_NOT_NULL(dst_ternsor);
  }
  if (place == tensor->place() && (!dst_ternsor->IsSharedWith(*tensor))) {
    dst_ternsor->ShareDataWith(*tensor);
  } else if (place != tensor->place()) {
    dst_ternsor->Resize(tensor->dims());
    dst_ternsor->mutable_data(place, tensor->dtype());
    paddle::framework::TensorCopySync(*tensor, place, dst_ternsor.get());
  }
  return dst_ternsor;
}

TensorPtr CreateTensorLike(const Tensor* tensor, const Place& place) {
  auto tensor_ptr = std::make_shared<Tensor>();
  PADDLE_ENFORCE_NOT_NULL(tensor_ptr);
  tensor_ptr->Resize(tensor->dims());
  tensor_ptr->mutable_data(place, tensor->dtype());
  return tensor_ptr;
}
}  // namespace

class SingleOpGcuExecutorImpl {
 public:
  void RunGcuOp(const std::vector<const Tensor*>& inputs,
                const std::vector<Tensor*>& outputs,
                const platform::Place ctx_place,
                const std::string& program_key,
                const int train_flag,
                const framework::Scope* curr_scope) {
    // std::vector<GcuTransInfo> trans_infos;
    std::vector<std::future<void>> vector_future;
    Recorder recorder(kProfilerValue);
    if (curr_scope != nullptr) {
      ResetScope(curr_scope);
    }
    auto begin = Now();
    auto start = begin;
    VLOG(2) << "=== Start run single op on Paddle 2.5 ===";
    ++train_iters_;
    std::call_once(init_once_, [&]() {
      start = Now();
      RunInit(ctx_place, program_key, train_flag);
      recorder.time_init = recorder.Cost(start, Now());
    });

    start = Now();
    VLOG(3) << "=== UpdateMemorys ===";
    UpdateMemorys(inputs, outputs);
    recorder.time_update_memory = recorder.Cost(start, Now());

    VLOG(3) << "is_train:" << is_train_ << " running_mode:" << running_mode_;

    start = Now();
    VLOG(3) << "=== RunExecutableAsync ===";

    PADDLE_GCU_TRACE_START(EXEC, exec);
    stream_->RunExecutableAsync(executable_, dev_inputs_, dev_outputs_);
    stream_->Synchronize();
    PADDLE_GCU_TRACE_END(EXEC, exec);

    recorder.time_executable_run = recorder.Cost(start, Now());

    if (is_train_) {
      // update weight in dev
      start = Now();
      VLOG(3) << "=== UpdateRefTensor ===";
      UpdateRefTensor();
      recorder.time_weights_post_process = recorder.Cost(start, Now());
    }

    recorder.time_total = recorder.Cost(begin, Now());
    if (recorder.IsEnableProfiler()) {
      VLOG(0) << std::endl
              << "=== Gcu Exec Statistics Info(Unit:ms) ===" << std::endl
              << "            iterator:" << train_iters_ << std::endl
              << "          total time:" << recorder.time_total << std::endl
              << "            init    :" << recorder.time_init << std::endl
              << "    update memory:   " << recorder.time_update_memory
              << std::endl
              << "      executable run:" << recorder.time_executable_run
              << std::endl
              << "weights post process:" << recorder.time_weights_post_process
              << std::endl;
    }
    VLOG(2) << "===  Run gcu_runtime_op success ===";
  }

  void RunInit(const platform::Place ctx_place,
               const std::string& program_key,
               const int train_flag) {
    VLOG(5) << "===  Run gcu_runtime_op start init ===";
    VLOG(5) << "GcuExecutor run program:" << program_key
            << ", train_flag:" << train_flag;
    program_key_ = program_key;
    global_mem_ref_ = TransformUtil::GetGlobalMemoryRef(program_key_);
    auto executables = TransformUtil::GetGcuExecutable(program_key_);
    PADDLE_ENFORCE_EQ(
        executables.size(),
        1,
        platform::errors::PreconditionNotMet(
            "Executables size:%zu should equals to 1.", executables.size()));

    executable_ = executables[0];
    PADDLE_ENFORCE_NOT_NULL(executable_);
    is_train_ = (train_flag == 1);
    running_mode_ = global_mem_ref_.running_mode;
    leaf_output_ = !(global_mem_ref_.leaf_outputs.empty());

    const auto input_keys = global_mem_ref_.input_keys.at(kSingleOpGraphId);
    const auto output_keys = global_mem_ref_.output_keys.at(kSingleOpGraphId);
    dev_inputs_.reserve(input_keys.size());
    dev_input_types_.reserve(input_keys.size());
    dev_input_names_.reserve(input_keys.size());

    dev_outputs_.resize(output_keys.size(), nullptr);
    dev_output_types_.resize(output_keys.size());
    dev_output_names_.resize(output_keys.size());

    AllocOutputTensors(executable_);
    VLOG(1) << "Finish RunInit, place:" << place_ << ", ctx place:" << ctx_place
            << ", is_train_:" << is_train_ << ", leaf_output_:" << leaf_output_
            << ", running_mode_:" << running_mode_
            << ", inputs key size:" << input_keys.size()
            << ", outputs size:" << dev_outputs_.size();
  }

  // *****************
  //   update memory
  // *****************
  void UpdateMemorys(const std::vector<const Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs) {
    GetWeightsFromScope();
    UpdateDataMemory(inputs);
    UpdateOutMemory(outputs);
    ReadjustRefOutputsMap();
    if (leaf_output_) {
      InitLeafOutputsInScope();
    }
  }
  void UpdateDataMemory(const std::vector<const Tensor*>& inputs) {
    VLOG(6) << "Update weight memory inputs size: " << inputs.size();
    if (inputs.empty()) return;
    VLOG(6) << "Update weight memory for eager mode.";
    const auto input_keys = global_mem_ref_.input_keys.at(kSingleOpGraphId);
    const auto var_to_symbol = global_mem_ref_.var_to_symbol;
    PADDLE_ENFORCE_GE(input_keys.size(),
                      inputs.size(),
                      platform::errors::PreconditionNotMet(
                          "input_keys size should greater or equals to "
                          "input's size"));
    int input_count = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto symbol = var_to_symbol.at(input_keys[i]);
      auto addr = inputs[i]->data();
      if (inputs[i]->place() != place_) {
        auto tmp_tensor = CreateTensorSharedOrFrom(
            inputs[i], place_, symbol_to_memory_[symbol]);
        symbol_to_memory_[symbol] = tmp_tensor;
        addr = tmp_tensor->data();
      }
      dev_inputs_.emplace_back(const_cast<void*>(addr));
      dev_input_types_.emplace_back(inputs[i]->dtype());
      dev_input_names_.emplace_back(symbol);

      VLOG(6) << "Update inputs[" << input_count << "] " << symbol << ", from "
              << inputs[i]->place() << ", addr:" << inputs[i]->data() << " to "
              << place_ << ", addr:" << addr << " dims: " << inputs[i]->dims();
      ++input_count;
    }

    for (size_t i = inputs.size(); i < input_keys.size(); ++i) {
      auto symbol = var_to_symbol.at(input_keys[i]);
      PADDLE_ENFORCE_NE(
          symbol_to_memory_.count(symbol),
          0,
          platform::errors::NotFound(
              "Failed to find weight memory, index:%zu, symbol:%s",
              i,
              symbol.c_str()));
      auto tensor = symbol_to_memory_.at(symbol);
      if (!(tensor->initialized() && tensor->numel() > 0)) {
        continue;
      }
      dev_inputs_.emplace_back(tensor->data());
      dev_input_types_.emplace_back(tensor->dtype());
      dev_input_names_.emplace_back(symbol);

      VLOG(6) << "Update inputs[" << input_count << "] " << symbol
              << ", location:" << tensor->place() << ", addr:" << tensor->data()
              << " dims: " << tensor->dims()
              << " initialized: " << tensor->initialized();
      ++input_count;
    }
  }

  void UpdateOutMemory(const std::vector<Tensor*>& outputs) {
    VLOG(6) << "Start update out memory for gcu runtime op, outputs num:"
            << outputs.size();
    const auto var_to_symbol = global_mem_ref_.var_to_symbol;
    const auto output_keys = global_mem_ref_.output_keys.at(kSingleOpGraphId);
    PADDLE_ENFORCE_GE(output_keys.size(),
                      outputs.size(),
                      platform::errors::PreconditionNotMet(
                          "Output_keys size should greater or equals to "
                          "output's size"));
    for (size_t i = 0; i < outputs.size(); ++i) {
      PADDLE_ENFORCE_NE(
          outputs[i], nullptr, platform::errors::NotFound("outputs is null"));
      auto* tensor = outputs[i];
      auto symbol = var_to_symbol.at(output_keys[i]);

      if (!(tensor->initialized())) {
        TensorPtr tmp_tensor = nullptr;
        if (symbol_to_memory_.count(symbol) > 0) {
          tmp_tensor = symbol_to_memory_.at(symbol);
        } else {
          // alloc for output
          tmp_tensor = std::make_shared<Tensor>(*(output_tensors_[i]));
          tmp_tensor->mutable_data(place_);
          symbol_to_memory_[symbol] = tmp_tensor;
        }
        tensor->ShareDataWith(*tmp_tensor);
      }
      dev_outputs_[i] = tensor->data();
      dev_output_types_[i] = tensor->dtype();
      dev_output_names_[i] = symbol;
      VLOG(6) << "Update outputs[" << i << "] " << symbol
              << ", to addr:" << dev_outputs_[i] << ", capacity is "
              << tensor->capacity() << ", type:" << tensor->dtype()
              << ", place:" << tensor->place()
              << ", ddim:" << tensor->dims().to_str();
    }
  }

  void UpdateRefTensor() {
    if (ref_output_update_.empty()) {
      return;
    }
    VLOG(6) << "UpdateRefTensor: start to update ref tensor";
    for (const auto& symbol_to_tensor_update : ref_output_update_) {
      auto tensor_src = symbol_to_tensor_update.second.first;
      Tensor* tensor_dst = symbol_to_tensor_update.second.second;
      if (tensor_dst == nullptr) {
        continue;
      }
      paddle::framework::TensorCopySync(
          *tensor_src, tensor_dst->place(), tensor_dst);

      VLOG(6) << "UpdateRefTensor for " << symbol_to_tensor_update.first
              << ", update " << tensor_dst->data() << " with "
              << tensor_src->data();
    }
    ref_output_update_.clear();
  }

  void InitLeafOutputsInScope() {
    VLOG(6) << "Start to init leaf outputs in scope";
    const auto leaf_outputs = global_mem_ref_.leaf_outputs;
    const auto leaf_output_keys = global_mem_ref_.leaf_output_keys;
    const auto var_to_symbol = global_mem_ref_.var_to_symbol;
    PADDLE_ENFORCE_EQ(leaf_outputs.size(),
                      leaf_output_keys.size(),
                      platform::errors::PreconditionNotMet(
                          "leaf outputs size should equals to key's size"));

    for (size_t i = 0; i < leaf_outputs.size(); ++i) {
      auto var_name = leaf_outputs[i];
      auto var = scope_->FindVar(var_name);
      if (var == nullptr) {
        continue;
      }
      auto symbol = var_to_symbol.at(leaf_output_keys[i]);
      size_t cnt = symbol_to_memory_.count(symbol);
      PADDLE_ENFORCE_NE(cnt,
                        0,
                        platform::errors::PreconditionNotMet(
                            "Device memory should be already "
                            "allocated, symbol:%s",
                            symbol.c_str()));
      auto gcu_tensor = symbol_to_memory_[symbol];
      auto var_tensor = var->GetMutable<phi::DenseTensor>();
      if (var_tensor->initialized()) {
        VLOG(6) << "Copy var tensor:" << var_name << " from "
                << var_tensor->place() << " to " << gcu_tensor->place();
        paddle::framework::TensorCopySync(
            *var_tensor, place_, gcu_tensor.get());
        var_tensor->ShareDataWith(*gcu_tensor);
      } else {
        var_tensor->ShareDataWith(*gcu_tensor);
        VLOG(6) << "Init var tensor:" << var_name << " to "
                << var_tensor->place() << ", addr:" << var_tensor->data();
      }
    }
  }

  // *****************
  //   alloc mem
  // *****************
  TensorPtr CreateDeviceTensor(const std::string& symbol,
                               const PaddleVarDesc& var_desc,
                               bool with_memory = true) {
    TensorPtr tensor = nullptr;
    if (!with_memory) {
      VLOG(6) << "CreateDeviceTensorWithMemory, symbol:" << symbol;
      tensor = std::make_shared<Tensor>();
      tensor->Resize(phi::make_ddim(var_desc.shapes));
      tensor->set_type(framework::TransToPhiDataType(var_desc.data_type));
      return tensor;
    }
    tensor = CreateTensor(place_, var_desc);
    VLOG(6) << "CreateDeviceTensor, symbol:" << symbol
            << ", addr:" << tensor->data();
    return tensor;
  }

  void AllocOutputTensors(const ExecutablePtr& exec) {
    VLOG(6) << "Start alloc output tensors";
    if (exec == nullptr) {
      return;
    }
    const auto var_to_symbol = global_mem_ref_.var_to_symbol;
    const auto output_keys = global_mem_ref_.output_keys.at(kSingleOpGraphId);
    auto resources = TransformUtil::GetExecutableRelections(exec);
    auto map_outputs_to_pd_var = resources.map_outputs_to_pd_var;
    for (size_t i = 0; i < map_outputs_to_pd_var.size(); ++i) {
      auto pd_var_desc = map_outputs_to_pd_var[i];
      auto symbol = var_to_symbol.at(output_keys.at(i));
      auto out = CreateDeviceTensor(symbol, pd_var_desc, false);
      output_tensors_.emplace_back(out);
      VLOG(6) << "alloc output tensor, symbol:" << symbol;
    }
    auto map_ref_out_to_weight = resources.map_ref_out_to_weight;
    auto map_inputs_to_pd_var = resources.map_inputs_to_pd_var;
    uint64_t exec_output_count = exec->NumOfOutputs();
    uint64_t exec_output_size[exec_output_count] = {0};
    exec->OutputSizeList(exec_output_size);
    PADDLE_ENFORCE_EQ(
        dev_outputs_.size(),
        exec_output_count,
        platform::errors::PreconditionNotMet(
            "dev_outputs_ size should greater or equals to exec_output_count"));
    for (size_t i = map_outputs_to_pd_var.size(); i < exec_output_count; ++i) {
      int32_t counter = map_ref_out_to_weight.count(i);
      PADDLE_ENFORCE_NE(
          counter,
          0,
          platform::errors::NotFound(
              "can not find the %u output correspond input info!Please check!",
              i));
      auto run_info = map_ref_out_to_weight[i];
      uint64_t output_size = exec_output_size[i];
      auto input_size = std::get<1>(run_info);
      auto input_idx = std::get<0>(run_info);
      PADDLE_ENFORCE_EQ(
          output_size,
          input_size,
          platform::errors::NotFound("the %zu output size[%llu] is not same "
                                     "with the %zu input size[%llu]",
                                     i,
                                     output_size,
                                     input_idx,
                                     input_size));
      auto cnt = map_inputs_to_pd_var.count(input_idx);
      PADDLE_ENFORCE_NE(
          cnt,
          0,
          platform::errors::NotFound(
              "the %zu output not found info on the index %zu of input map",
              i,
              input_idx));
      auto inputs_to_pd_var = map_inputs_to_pd_var[input_idx];
      auto symbol = var_to_symbol.at(output_keys.at(i));
      auto extra = CreateDeviceTensor(symbol, inputs_to_pd_var, true);
      output_tensors_.emplace_back(extra);
      persistent_tensors_.emplace(symbol, extra);
      dev_outputs_[i] = extra->data();
      dev_output_types_[i] = extra->dtype();
      dev_output_names_[i] = symbol;
      VLOG(6) << "alloc extra output Memory, symbol:" << symbol
              << ", addr:" << dev_outputs_[i];
    }
  }

  void ReadjustRefOutputsMap() {
    VLOG(6) << "Readjust ref outputs map for eager mode.";
    const auto weight_update_params = global_mem_ref_.weight_update_params;
    const auto weight_to_symbol = global_mem_ref_.weight_to_symbol;
    for (const auto& update_param : weight_update_params) {
      auto var_name = update_param.first;
      auto param = update_param.second;
      size_t cnt = weight_to_symbol.count(var_name);
      PADDLE_ENFORCE_NE(cnt,
                        0,
                        platform::errors::NotFound(
                            "var should have symbol in weight_to_symbol, "
                            "var_name:%s",
                            var_name.c_str()));
      auto weight_symbol = weight_to_symbol.at(var_name);
      auto src_symbol = param.symbol;
      cnt = symbol_to_memory_.count(weight_symbol);
      PADDLE_ENFORCE_NE(
          cnt,
          0,
          platform::errors::NotFound("Device memory should be already "
                                     "allocated, var_name:%s, weight_symbol:%s",
                                     var_name.c_str(),
                                     weight_symbol.c_str()));
      cnt = persistent_tensors_.count(src_symbol);
      PADDLE_ENFORCE_NE(
          cnt,
          0,
          platform::errors::NotFound("Device memory should be already "
                                     "allocated, var_name:%s, src_symbol:%s",
                                     var_name.c_str(),
                                     src_symbol.c_str()));
      auto& weight_tensor = symbol_to_memory_[weight_symbol];
      auto& src_weight = persistent_tensors_[src_symbol];
      ref_output_update_[var_name] =
          std::make_pair(src_weight, weight_tensor.get());
      VLOG(6) << "ReadjustRefOutput " << var_name << " as: ["
              << weight_tensor->place() << ", addr:" << weight_tensor->data()
              << "] will be updated by [" << src_weight->place()
              << ", addr:" << src_weight->data() << "].";
    }
  }

  void GetWeightsFromScope() {
    const auto weights = global_mem_ref_.weights;
    const auto weight_to_symbol = global_mem_ref_.weight_to_symbol;
    for (const auto var_name : weights) {
      auto var = scope_->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          var, "Failed to find %s in scope.", var_name.c_str());
      auto var_tensor = var->GetMutable<LoDTensor>();
      auto symbol = weight_to_symbol.at(var_name);
      if (var_tensor->numel() <= 0 && !(var_tensor->initialized())) {
        auto tensor = CreateTensorLike(var_tensor, place_);
        symbol_to_memory_[symbol] = tensor;
        VLOG(6) << "GetWeightsFromScope, var name:" << var_name
                << " numel is 0, placehold addr:" << tensor->data();
        continue;
      }
      PADDLE_ENFORCE_EQ(
          var_tensor->initialized(),
          true,
          platform::errors::PreconditionNotMet(
              "Weight in scope should already been initialized, name:%s",
              var_name.c_str()));
      auto dst_tensor = symbol_to_memory_[symbol];  // Get or add nullptr
      auto tensor = CreateTensorSharedOrFrom(var_tensor, place_, dst_tensor);
      symbol_to_memory_[symbol] = tensor;
      if (var_tensor->place() != tensor->place()) {
        VLOG(6) << "GetWeightsFromScope, transform " << var_name << " from "
                << var_tensor->place() << " to " << tensor->place();
        var_tensor->ShareDataWith(*tensor);
      } else {
        VLOG(6) << "GetWeightsFromScope, var name:" << var_name
                << " locates on:" << tensor->place()
                << ", addr:" << tensor->data();
      }
    }
  }

  explicit SingleOpGcuExecutorImpl(const framework::Scope* scope) {
    scope_ = scope;
    int device_id = runtime::GcuGetCurrentDevice();
    place_ = CustomPlace("gcu", device_id);
    ctx_ = runtime::GcuGetContext(device_id);
    PADDLE_ENFORCE_NE(
        ctx_, nullptr, platform::errors::NotFound("create runtime ctx failed"));
    stream_ = ctx_->default_exe_stream;
    PADDLE_ENFORCE_NE(
        stream_, nullptr, platform::errors::NotFound("create stream failed"));

    VLOG(1) << "Init SingleOpGcuExecutorImpl for device_id:" << device_id
            << ", place:" << place_;
  }

  void ReleaseResource() {
    ReleaseAllMemory();
    executable_ = nullptr;
    TransformUtil::GraphToGcuExecutable(program_key_, {}, {});
  }

  void ReleaseMemory() {
    ref_output_update_.clear();
    symbol_to_memory_.clear();
  }

  void ReleaseAllMemory() {
    // GCU memory is managed uniformly in symbol_to_memory_ and
    // persistent_tensors_
    ReleaseMemory();
    output_tensors_.clear();
    persistent_tensors_.clear();
  }

  void ResetScope(const framework::Scope* scope) { scope_ = scope; }

  SingleOpGcuExecutorImpl(const SingleOpGcuExecutorImpl& impl) = default;

  ~SingleOpGcuExecutorImpl() {}

  SingleOpGcuExecutorImpl& operator=(const SingleOpGcuExecutorImpl& impl) =
      default;

 private:
  std::once_flag init_once_;
  GcuCtxPtr ctx_ = nullptr;
  std::string program_key_;
  GcuStreamPtr stream_ = nullptr;
  CustomPlace place_;
  const framework::Scope* scope_ = nullptr;
  ExecutablePtr executable_;
  GlobalMemoryRef global_mem_ref_;
  std::vector<void*> dev_inputs_;
  std::vector<void*> dev_outputs_;
  std::vector<DataType> dev_input_types_;
  std::vector<DataType> dev_output_types_;
  std::vector<std::string> dev_input_names_;
  std::vector<std::string> dev_output_names_;
  TensorPtr tmp_output_tensor_ = nullptr;
  std::unordered_map<std::string, TensorPtr> symbol_to_memory_;
  std::vector<TensorPtr> output_tensors_;
  std::unordered_map<std::string, TensorPtr> persistent_tensors_;
  // output_symbol, <src_tensor, dst_tensor>
  std::unordered_map<std::string, std::pair<TensorPtr, Tensor*>>
      ref_output_update_;
  bool is_train_ = true;
  bool leaf_output_ = false;
  std::string running_mode_ = RunningMode::SERIAL;
  int64_t train_iters_ = 0;
};

SingleOpGcuExecutor::SingleOpGcuExecutor(const framework::Scope* scope) {
  impl_ = std::make_shared<SingleOpGcuExecutorImpl>(scope);
}

void SingleOpGcuExecutor::ReleaseResource() { impl_->ReleaseResource(); }

void SingleOpGcuExecutor::ReleaseMemory() { impl_->ReleaseMemory(); }

void SingleOpGcuExecutor::ResetScope(const framework::Scope* scope) {
  impl_->ResetScope(scope);
}

void SingleOpGcuExecutor::RunGcuOp(const std::vector<const Tensor*>& inputs,
                                   const std::vector<Tensor*>& outputs,
                                   const platform::Place ctx_place,
                                   const std::string& program_key,
                                   const int train_flag,
                                   const framework::Scope* curr_scope) {
  impl_->RunGcuOp(
      inputs, outputs, ctx_place, program_key, train_flag, curr_scope);
}

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
