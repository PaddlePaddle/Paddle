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

#include "paddle/fluid/platform/device/gcu/executor/gcu_executor.h"

#include <algorithm>
#include <chrono>  // NOLINT [build/c++11]
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>  // NOLINT [build/c++11]
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "dtu/hlir/builder/hlir_builder.h"
#include "gcu/umd/dtu_assembler_def.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gcu/common/gcu_options.h"
#include "paddle/fluid/platform/device/gcu/gcu_backend.h"
#include "paddle/fluid/platform/device/gcu/profile/profile.h"
#include "paddle/fluid/platform/device/gcu/runtime/gcu_rt_interface.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_executable.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace platform {
namespace gcu {
using GcuGraphPtr = std::shared_ptr<hlir::Module>;
using ProcessGroupPtr = std::shared_ptr<paddle::distributed::ProcessGroup>;
using GcuCtxPtr = paddle::platform::gcu::runtime::GcuCtxPtr;
using GcuStreamPtr = paddle::platform::gcu::runtime::GcuStreamPtr;
using ExecutablePtr = paddle::platform::gcu::runtime::ExecutablePtr;
using Place = paddle::platform::Place;
using CustomPlace = paddle::platform::CustomPlace;
using DDim = paddle::framework::DDim;
using TensorPtr = std::shared_ptr<phi::DenseTensor>;
using paddle::framework::ExecutionContext;
using paddle::framework::Scope;
using paddle::framework::ir::Graph;
using paddle::framework::ir::Node;
using paddle::platform::gcu::kGcuProgramKey;
using phi::DenseTensor;

using namespace std;          // NOLINT
using namespace std::chrono;  // NOLINT

namespace {
const char* const kTensorFusion = "PADDLE_GCU_COLLECTIVE_TENSOR_FUSION";
const char* const kProfilerValue = std::getenv(kProfiler);
const char* const kTensorFusionValue = std::getenv(kTensorFusion);
const char* const kMemoryTypeCollective = "Collective";
const char* const kMemoryTypeTransWeights = "TransWeights";
const char* const kMemoryTypeTransWeightsSrc = "TransWeightsSrc";
const char* const kMemoryTypeTransWeightsDst = "TransWeightsDst";
const char* const kMemoryTypeWeights = "Weights";
const size_t kGlobalGraphID = 0;
const size_t kFpBpGraphID = 1;
const size_t kUpdateGraphID = 2;
const uint32_t kGlobalGroupID = 0;
const size_t kCollectiveParamsInput = 0;
const size_t kCollectiveParamsOutput = 1;
constexpr int64_t kMaxFusionMemSize = 2 * 1024 * 1024 * 1024L;  // 2GB
const int64_t kMemAlignSize = 128;

GcuThreadPool g_thread_pool;

// funs for debug
std::string StringVectorDebugStr(const std::vector<std::string>& strs,
                                 const std::string& debug_info = "") {
  std::ostringstream os;
  os << "debug info:" << debug_info << "\n";
  for (const auto& str : strs) {
    os << "    value:" << str << "\n";
  }
  return os.str();
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

TensorPtr CreateTensor(const TensorPtr& tensor,
                       const PaddleVarDesc& var_desc,
                       size_t offset) {
  auto sub_tensor_ptr = std::make_shared<Tensor>();
  PADDLE_ENFORCE_NOT_NULL(sub_tensor_ptr);
  sub_tensor_ptr->ShareDataWith(*tensor);
  sub_tensor_ptr->Resize(phi::make_ddim(var_desc.shapes));
  sub_tensor_ptr->set_offset(offset);
  return sub_tensor_ptr;
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

TensorPtr CreateTensorLike(const TensorPtr& tensor, const Place& place) {
  auto tensor_ptr = std::make_shared<Tensor>();
  PADDLE_ENFORCE_NOT_NULL(tensor_ptr);
  tensor_ptr->Resize(tensor->dims());
  tensor_ptr->mutable_data(place, tensor->dtype());
  return tensor_ptr;
}
}  // namespace

class GcuExecutorImpl {
 public:
  void RunGcuOp(const std::vector<const Tensor*>& inputs,
                const std::vector<Tensor*>& outputs,
                const paddle::framework::ExecutionContext& ctx,
                const std::string& program_key,
                const int train_flag,
                const framework::Scope* curr_scope) {
    if (tmp_input_tensor_ == nullptr) {
      tmp_input_tensor_ = std::make_shared<Tensor>(Tensor());
      tmp_input_tensor_->Resize(phi::make_ddim({1}));
      tmp_input_tensor_->mutable_data(ctx.GetPlace(), DataType::FLOAT32);

      Tensor tmp_cpu;
      tmp_cpu.Resize(phi::make_ddim({1}));
      tmp_cpu.mutable_data(CPUPlace(), DataType::FLOAT32);
      framework::TensorCopy(tmp_cpu, ctx.GetPlace(), tmp_input_tensor_.get());
    }
    if (tmp_output_tensor_ == nullptr) {
      tmp_output_tensor_ = std::make_shared<Tensor>(Tensor());
      tmp_output_tensor_->Resize(phi::make_ddim({1}));
      tmp_output_tensor_->mutable_data(ctx.GetPlace(), DataType::FLOAT32);

      Tensor tmp_cpu;
      tmp_cpu.Resize(phi::make_ddim({1}));
      tmp_cpu.mutable_data(CPUPlace(), DataType::FLOAT32);
      framework::TensorCopy(tmp_cpu, ctx.GetPlace(), tmp_output_tensor_.get());
    }

    // std::vector<GcuTransInfo> trans_infos;
    std::vector<std::future<void>> vector_future;
    Recorder recorder(kProfilerValue);
    if (curr_scope != nullptr) {
      ResetScope(curr_scope);
    }
    auto begin = system_clock::now();
    auto start = begin;
    VLOG(2) << "=== Start run gcu_runtime_op on Paddle 2.5 ===";
    ++train_iters_;
    std::call_once(init_once_, [&]() {
      start = system_clock::now();
      RunInit(ctx, program_key, train_flag);
      recorder.time_init = recorder.Cost(start, system_clock::now());
    });

    start = system_clock::now();
    auto start_weight_inv_trans = start;
    VLOG(3) << "=== UpdateDataMemory ===";
    // update feed addrs
    UpdateDataMemory(inputs);
    recorder.time_update_input_memory =
        recorder.Cost(start, system_clock::now());

    start = system_clock::now();
    VLOG(3) << "=== UpdateOutMemory ===";
    // update fetch addrs
    UpdateOutMemory(outputs);
    recorder.time_update_output_memory =
        recorder.Cost(start, system_clock::now());

    VLOG(3) << "is_train:" << is_train_ << " weight_init:" << weight_init_
            << " running_mode:" << running_mode_;

    if (is_train_ && weight_init_ && running_mode_ == RunningMode::ADAPTIVE &&
        weight_sync_smoothly_ && (!no_need_trans_weights_)) {
      VLOG(3) << "=== HostInvTransWeight ===";
      start_weight_inv_trans = system_clock::now();
      UpdateWeightsDevice();
      InverseTransWeightsInHost(vector_future);
    }

    if (!weight_init_) {
      // Broadcast here
      if (is_distributed_) {
        start = system_clock::now();
        VLOG(3) << "=== BroadcastWeights ===";
        BroadcastWeights();
        recorder.time_dist_brd_weights =
            recorder.Cost(start, system_clock::now());
      }
      weight_init_ = true;
    }

    start = system_clock::now();
    for (size_t i = 0; i < executables_.size(); ++i) {
      ExecutablePtr exec = executables_[i];
      if (exec == nullptr) {
        continue;
      }
      VLOG(3) << "=== RunExecutableAsync graph:" << i << " ===";
      stream_->RunExecutableAsync(exec, dev_inputs_[i], dev_outputs_[i]);

      // sync for run
      stream_->Synchronize();

      if (is_distributed_ && i == kFpBpGraphID) {
        auto start = system_clock::now();
        VLOG(3) << "=== AllreduceGrads ===";
        // allreduce here
        AllreduceGrads();
        recorder.time_dist_allreduce =
            recorder.Cost(start, system_clock::now());
      }
    }
    recorder.time_executable_run = recorder.Cost(start, system_clock::now());

    if (is_train_ && weight_init_ && running_mode_ == RunningMode::ADAPTIVE &&
        weight_sync_smoothly_ && (!no_need_trans_weights_)) {
      // wait async transpose finish
      std::for_each(vector_future.begin(),
                    vector_future.end(),
                    [](const std::future<void>& f) { f.wait(); });
      TransWeightsToDevice();
      recorder.time_weights_inv_trans =
          recorder.Cost(start_weight_inv_trans, system_clock::now());
    }

    if (is_train_) {
      // update weight in dev
      start = system_clock::now();
      if (running_mode_ == RunningMode::ADAPTIVE && weight_sync_smoothly_ &&
          (!no_need_trans_weights_)) {
        // do nothing
        // update in the next begining, before InverseTransWeightsInHost
      } else {
        VLOG(3) << "=== UpdateWeightsDevice ===";
        UpdateWeightsDevice();
      }
      recorder.time_weights_post_process =
          recorder.Cost(start, system_clock::now());
    }

    if ((sync_interval_ != 1) && (train_iters_ % sync_interval_ == 0)) {
      start = system_clock::now();
      Synchronize();
      recorder.time_sync = recorder.Cost(start, system_clock::now());
    }

    IteratorSync();
    recorder.time_total = recorder.Cost(begin, system_clock::now());
    if (recorder.IsEnableProfiler()) {
      VLOG(0) << endl
              << "=== Gcu Exec Statistics Info(Unit:ms) ===" << endl
              << "            iterator:" << train_iters_ << endl
              << "          total time:" << recorder.time_total << endl
              << "            init    :" << recorder.time_init << endl
              << " update input memory:" << recorder.time_update_input_memory
              << endl
              << "updateoutnput memory:" << recorder.time_update_output_memory
              << endl
              << "  weights inv trans :" << recorder.time_weights_inv_trans
              << endl
              << "distribute broadcast:" << recorder.time_dist_brd_weights
              << endl
              << "      executable run:" << recorder.time_executable_run << endl
              << "distribute allreduce:" << recorder.time_dist_allreduce << endl
              << "weights post process:" << recorder.time_weights_post_process
              << endl
              << "        weights sync:" << recorder.time_sync << endl;
    }
    VLOG(2) << "===  Run gcu_runtime_op success ===";
  }

  void RunInit(const paddle::framework::ExecutionContext& ctx,
               const std::string& program_key,
               const int train_flag) {
    VLOG(5) << "===  Run gcu_runtime_op start init ===";
    VLOG(5) << "GcuExecutor run program:" << program_key
            << ", train_flag:" << train_flag;
    program_key_ = program_key;
    executables_ = TransformUtil::GetGcuExecutable(program_key_);
    global_mem_ref_ = TransformUtil::GetGlobalMemoryRef(program_key_);
    if (train_flag <= 0) {
      is_train_ = global_mem_ref_.is_training_graph;
    } else {
      is_train_ = (train_flag == 1);
    }
    running_mode_ = global_mem_ref_.running_mode;
    is_distributed_ = is_distributed_ && is_train_;
    leaf_output_ = !(global_mem_ref_.leaf_outputs.empty());

    if (kTensorFusionValue != nullptr &&
        (std::string(kTensorFusionValue) == "false")) {
      collective_tensor_fusion_ = false;
    }

    std::string weight_sync_option =
        platform::gcu::GetGcuOptions().GetOption("gcu_weights_sync_mode");

    std::string sync_interval_option =
        platform::gcu::GetGcuOptions().GetOption("gcu_weights_sync_interval");
    if (!sync_interval_option.empty()) {
      sync_interval_ = TransformUtil::StringToNumber(sync_interval_option);
    }

    // Do not directly update weights to host in one of the following cases:
    // 1. The user has set the synchronization interval
    // 2. The user has set the manual synchronization mode
    bool not_directly_update =
        (sync_interval_ != 1) || (weight_sync_option == "manually");
    weight_sync_smoothly_ = !not_directly_update;

    GenerateGlobalSymbol();
    AllocMemory(executables_);
    GenerateGlobalMem();
    if (is_train_) {
      InitNonTrainableWeights();
    }
    if (leaf_output_) {
      InitLeafOutputsInScope();
    }
    if (is_train_ && running_mode_ == RunningMode::ADAPTIVE) {
      AllocMemoryForWeightTransfer();
      TransferMemoryForInitWeights();
    }
    VLOG(1) << "Finish RunInit, place:" << place_
            << ", ctx place:" << ctx.GetPlace() << ", is_train_:" << is_train_
            << ", is_distributed_:" << is_distributed_
            << ", leaf_output_:" << leaf_output_
            << ", running_mode_:" << running_mode_
            << ", weights need to trans:" << var_to_transed_info_.size()
            << ", no_need_trans_weights_:" << no_need_trans_weights_
            << ", sync_interval_:" << sync_interval_
            << ", weight_sync_smoothly_:" << weight_sync_smoothly_;
  }

  // *****************
  //   update memory
  // *****************
  void UpdateDataMemory(const std::vector<const Tensor*>& inputs) {
    if (inputs.empty()) return;
    PADDLE_ENFORCE_EQ(
        inputs.size(),
        global_input_symbols_.size(),
        platform::errors::NotFound("Inputs size:%zu should equals to device's "
                                   "global_input_symbols_:%zu",
                                   inputs.size(),
                                   global_input_symbols_.size()));
    PADDLE_ENFORCE_EQ(inputs.size(), global_inputs_.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto symbol = global_input_symbols_[i];
      size_t cnt = global_input_to_dev_input_.count(symbol);
      PADDLE_ENFORCE_NE(cnt,
                        0,
                        platform::errors::NotFound(
                            "Failed to map data memory, index:%zu, symbol:%s",
                            i,
                            symbol.c_str()));

      auto& to_updates = global_input_to_dev_input_[symbol];
      for (const auto& item : to_updates) {
        size_t graph_id = item.first;
        size_t input_idx = item.second;
        auto addr = inputs[i]->data();
        if (inputs[i]->place() != place_) {
          global_inputs_[i] =
              CreateTensorSharedOrFrom(inputs[i], place_, global_inputs_[i]);
          symbol_to_memory_[symbol] = global_inputs_[i];
          addr = global_inputs_[i]->data();
        }
        dev_inputs_[graph_id][input_idx] = const_cast<void*>(addr);
        dev_input_types_[graph_id][input_idx] = inputs[i]->dtype();
        VLOG(6) << "UpdateDataMemory data palce:" << inputs[i]->place()
                << ", addr:" << inputs[i]->data() << " to " << place_
                << ", addr:" << addr << ", graph:" << graph_id
                << ", index:" << input_idx << ", global index:" << i;
      }
    }
  }

  void UpdateOutMemory(const std::vector<Tensor*>& outputs) {
    VLOG(6) << "Start update out memory for gcu runtime op, outputs num:"
            << outputs.size();
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        global_outputs_.size(),
        platform::errors::NotFound("Outputs size should equals to "
                                   "device's global_outputs_ size"));
    for (size_t i = 0; i < outputs.size(); i++) {
      PADDLE_ENFORCE_NE(
          outputs[i], nullptr, platform::errors::NotFound("outputs is null"));
      auto* tensor = outputs[i];
      tensor->ShareDataWith(*(global_outputs_[i]));
      VLOG(6) << "outputs[" << i << "] capacity is " << tensor->capacity()
              << ", type:" << tensor->dtype() << ", place:" << tensor->place()
              << ", ddim:" << tensor->dims().to_str();
    }
  }

  void BroadcastWeights() {
    VLOG(6) << "start to broadcast weights";
    std::vector<std::shared_ptr<paddle::distributed::ProcessGroup::Task>> tasks;
    for (const auto weight : total_weights_) {
      std::vector<Tensor> in_tensors{*(weight)};
      auto task = process_group_->Broadcast(in_tensors, in_tensors);
      tasks.emplace_back(task);
    }
    for (auto& task : tasks) {
      task->Wait();
    }
  }

  void AllreduceGrads() {
    VLOG(6) << "start to allreduce grads";
    paddle::distributed::AllreduceOptions opts;
    opts.reduce_op = paddle::distributed::ReduceOp::SUM;
    std::vector<std::shared_ptr<paddle::distributed::ProcessGroup::Task>> tasks;
    for (const auto allreduce_tensor : allreduce_tensors_) {
      std::vector<Tensor> in_tensors{*(allreduce_tensor.first)};
      std::vector<Tensor> output_tensors{*(allreduce_tensor.second)};
      auto task = process_group_->AllReduce(in_tensors, output_tensors, opts);
      tasks.emplace_back(task);
    }
    for (auto& task : tasks) {
      task->Wait();
    }
  }

  void InverseTransWeightsInHost(
      std::vector<std::future<void>>& vector_future) {  // NOLINT
    VLOG(6) << "start to inverse trans weights";
    std::vector<GcuTransInfo> trans_infos;
    const auto weights = global_mem_ref_.weights;
    for (const auto weight_name : weights) {
      if (var_to_transed_info_.count(weight_name) != 0) {
        trans_infos.emplace_back(var_to_transed_info_[weight_name]);
      }
    }
    TransWeightsToHost();
    auto trans_func = [](const GcuTransInfo& args) {
      // first inverse param
      GcuTransInfo trans_info;
      trans_info.dst_layout = args.srs_layout;
      trans_info.srs_layout = args.dst_layout;
      trans_info.dst_shape = args.src_shape;
      trans_info.src_shape = args.dst_shape;
      trans_info.element_bytes = args.element_bytes;
      trans_info.src_tensor = args.dst_tensor;
      trans_info.dst_tensor = args.src_tensor;
      trans_info.src_data = args.dst_data;
      trans_info.dst_data = args.src_data;

      // do transpose
      GcuTransfer transfer;
      transfer.Trans(trans_info, false);
    };

    for (const auto& arg : trans_infos) {
      std::future<void> f = g_thread_pool.commit(trans_func, arg);
      vector_future.emplace_back(std::move(f));
    }
  }

  void UpdateWeightsDevice() {
    VLOG(6) << "start to update weights in dev";
    for (const auto& name_to_weight_update : weights_update_) {
      auto weight_tensor = name_to_weight_update.second.first;
      auto update_tensor = name_to_weight_update.second.second;
      paddle::framework::TensorCopySync(
          *update_tensor, place_, weight_tensor.get());
      VLOG(6) << "UpdateWeightsDevice for " << name_to_weight_update.first
              << ", update " << weight_tensor->data() << " with "
              << update_tensor->data();
    }
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
      auto var_tensor = var->GetMutable<Tensor>();
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
  void AllocMemoryForWeightTransfer(int64_t align_size = kMemAlignSize) {
    VLOG(6) << "Start alloc memory for weights transfer";
    const auto global_transed_weights_info = global_mem_ref_.weights_trans_info;
    const auto weight_update_params = global_mem_ref_.weight_update_params;
    const auto weight_to_symbol = global_mem_ref_.weight_to_symbol;
    size_t fused_trans_weight_size =
        memory_stores_[kMemoryTypeTransWeights].size();
    VLOG(6) << "Global transed weights info size:"
            << global_transed_weights_info.size()
            << ", fused trans weight size:" << fused_trans_weight_size;
    if (fused_trans_weight_size == 0) {
      VLOG(6) << "No need to trans weights.";
      no_need_trans_weights_ = true;
      return;
    }

    PADDLE_ENFORCE_EQ(
        fused_trans_weight_size,
        2,
        platform::errors::PreconditionNotMet(
            "it is fused into one currently, %s num should be 2, but get %zu",
            kMemoryTypeTransWeights,
            fused_trans_weight_size));

    auto dst_fused_tensor = memory_stores_[kMemoryTypeTransWeights].at(0);
    auto src_fused_tensor =
        CreateTensorLike(dst_fused_tensor, dst_fused_tensor->place());
    memory_stores_[kMemoryTypeTransWeightsSrc].emplace_back(src_fused_tensor);
    memory_stores_[kMemoryTypeTransWeightsDst].emplace_back(dst_fused_tensor);

    auto src_fused_cpu =
        CreateTensorLike(src_fused_tensor, platform::CPUPlace());
    auto dst_fused_cpu =
        CreateTensorLike(dst_fused_tensor, platform::CPUPlace());
    memory_stores_[kMemoryTypeTransWeightsSrc].emplace_back(src_fused_cpu);
    memory_stores_[kMemoryTypeTransWeightsDst].emplace_back(dst_fused_cpu);

    for (const auto& name_to_trans_info : global_transed_weights_info) {
      auto var_name = name_to_trans_info.first;
      if (var_to_transed_info_.count(var_name) != 0) {
        continue;
      }
      size_t cnt = weight_update_params.count(var_name);
      if (cnt == 0) {
        continue;
      }
      cnt = weight_to_symbol.count(var_name);
      PADDLE_ENFORCE_NE(cnt,
                        0,
                        platform::errors::NotFound(
                            "var should have symbol in weight_to_symbol, "
                            "var_name:%s",
                            var_name.c_str()));
      auto symbol = weight_to_symbol.at(var_name);
      cnt = symbol_to_memory_.count(symbol);
      PADDLE_ENFORCE_NE(cnt,
                        0,
                        platform::errors::PreconditionNotMet(
                            "Device memory should be already "
                            "allocated, symbol:%s",
                            symbol.c_str()));
      auto info = name_to_trans_info.second;
      auto var_desc = weight_update_params.at(var_name).var_desc;
      auto dst_tensor = symbol_to_memory_[symbol];
      size_t offset = dst_tensor->offset();
      info.src_tensor = CreateTensor(src_fused_tensor, var_desc, offset);
      UpdateScopeWeightTensor(var_name, info.src_tensor);
      info.dst_tensor = dst_tensor;

      info.src_data =
          reinterpret_cast<uint8_t*>(src_fused_cpu->data()) + offset;
      info.dst_data =
          reinterpret_cast<uint8_t*>(dst_fused_cpu->data()) + offset;

      //   paddle::framework::TensorCopySync(*info.dst_tensor, place_,
      //                                     info.src_tensor.get());

      var_to_transed_info_[var_name] = info;

      VLOG(6) << "Alloc memory for trans " << var_name
              << ", src addr:" << info.src_tensor->data()
              << ", dst addr:" << info.dst_tensor->data();
    }
    paddle::framework::TensorCopySync(
        *dst_fused_tensor, place_, src_fused_tensor.get());
    paddle::framework::TensorCopySync(
        *src_fused_tensor, platform::CPUPlace(), src_fused_cpu.get());
    paddle::framework::TensorCopySync(
        *dst_fused_tensor, platform::CPUPlace(), dst_fused_cpu.get());
  }

  void UpdateScopeWeightTensor(const std::string& var_name,
                               const TensorPtr& tensor) {
    auto var = scope_->FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var, "Failed to find %s in scope.", var_name.c_str());
    auto var_tensor = var->GetMutable<Tensor>();
    PADDLE_ENFORCE_EQ(
        var_tensor->initialized(),
        true,
        platform::errors::PreconditionNotMet(
            "Weight in scope should already been initialized, name:%s",
            var_name.c_str()));
    var_tensor->ShareDataWith(*tensor);
    VLOG(6) << "UpdateScopeWeightTensor var:" << var_name
            << ", location:" << var_tensor->place()
            << ", addr:" << var_tensor->data();
  }

  void TransferMemoryForInitWeights() {
    if (no_need_trans_weights_) {
      return;
    }
    VLOG(6) << "Start transfer memory for init weights.";
    GcuTransfer transfer;
    for (const auto& var_to_info : var_to_transed_info_) {
      auto var_name = var_to_info.first;
      auto trans_info = var_to_info.second;
      if (trans_info.has_transed) {
        continue;
      }
      //   paddle::framework::TensorCopySync(*trans_info.dst_tensor, place_,
      //                                     trans_info.src_tensor.get());

      auto res_info = transfer.Trans(trans_info, false);
      res_info.has_transed = true;
      var_to_transed_info_[var_name] = res_info;
      VLOG(6) << "Transfer memory for init weight:" << var_name;
    }
    auto dst_gcu_tensor = memory_stores_[kMemoryTypeTransWeightsDst].at(0);
    auto dst_cpu_tensor = memory_stores_[kMemoryTypeTransWeightsDst].at(1);
    paddle::framework::TensorCopySync(
        *dst_cpu_tensor, place_, dst_gcu_tensor.get());
  }

  void AllocMemory(const std::vector<ExecutablePtr>& executables) {
    if (is_distributed_ && collective_tensor_fusion_) {
      const auto& allreduce_params = global_mem_ref_.allreduce_params;
      CollectiveTensorFusion(allreduce_params, kMemAlignSize);
    }
    if (is_train_) {
      WeightTensorFusion(kMemAlignSize);
    }

    // for inference to get var in scope
    if ((!is_train_) && (!leaf_output_)) {
      GetWeightsFromScope();
    }
    for (size_t graph_id = 0; graph_id < executables.size(); ++graph_id) {
      ExecutablePtr exec = executables[graph_id];
      AllocInputMemory(exec, graph_id);
      AllocOutputMemory(exec, graph_id);
    }
  }

  TensorPtr GetOrCreateDeviceTensor(const std::string& symbol,
                                    const PaddleVarDesc& var_desc) {
    TensorPtr tensor = nullptr;
    if (global_tensor_skip_alloc_.count(symbol) > 0) {
      VLOG(6) << "GetOrCreateDeviceTensor, skip global tensor, symbol:"
              << symbol;
      tensor = std::make_shared<Tensor>();
      symbol_to_memory_.emplace(symbol, tensor);
      return tensor;
    }
    if (symbol_to_memory_.count(symbol) != 0) {
      tensor = symbol_to_memory_[symbol];
      PADDLE_ENFORCE_GE(tensor->capacity(),
                        var_desc.data_size,
                        platform::errors::PreconditionNotMet(
                            "symbol %s mem size should grater or equals to "
                            "param var_desc data size",
                            symbol.c_str()));
      VLOG(6) << "Memory allready allocated, symbol:" << symbol
              << ", addr:" << tensor->data();
    } else {
      tensor = CreateTensor(place_, var_desc);
      symbol_to_memory_.emplace(symbol, tensor);
      VLOG(6) << "GetOrCreateDeviceTensor ADD symbol:" << symbol
              << ", addr:" << tensor->data();
    }
    return tensor;
  }

  void AllocInputMemory(const ExecutablePtr& exec, size_t graph_id) {
    VLOG(6) << "Start alloc input Memory, graph_id:" << graph_id;
    if (exec == nullptr) {
      return;
    }
    const auto var_to_symbol = global_mem_ref_.var_to_symbol;
    const auto input_keys = global_mem_ref_.input_keys.at(graph_id);
    auto info =
        "AllocInputMemory input_keys for graph:" + std::to_string(graph_id);
    VLOG(6) << StringVectorDebugStr(input_keys, info);
    auto resources = TransformUtil::GetExecutableRelections(exec);
    auto map_inputs_to_pd_var = resources.map_inputs_to_pd_var;
    for (size_t i = 0; i < map_inputs_to_pd_var.size(); ++i) {
      auto pd_var_desc = map_inputs_to_pd_var[i];
      VLOG(6) << pd_var_desc.var_name << " capacity is "
              << pd_var_desc.data_size;
      auto symbol = var_to_symbol.at(input_keys.at(i));
      auto input = GetOrCreateDeviceTensor(symbol, pd_var_desc);
      auto* addr = (input->initialized()) ? (input->data()) : nullptr;
      global_input_to_dev_input_[symbol].emplace_back(
          std::make_pair(graph_id, dev_inputs_[graph_id].size()));
      dev_inputs_[graph_id].emplace_back(addr);
      dev_input_types_[graph_id].emplace_back(input->dtype());
      VLOG(6) << "alloc input Memory, symbol:" << symbol << ", addr:" << addr;
    }
  }

  void AllocOutputMemory(const ExecutablePtr& exec, size_t graph_id) {
    VLOG(6) << "Start alloc output Memory, graph_id:" << graph_id;
    if (exec == nullptr) {
      return;
    }
    const auto var_to_symbol = global_mem_ref_.var_to_symbol;
    const auto output_keys = global_mem_ref_.output_keys.at(graph_id);
    auto resources = TransformUtil::GetExecutableRelections(exec);
    auto map_outputs_to_pd_var = resources.map_outputs_to_pd_var;
    for (size_t i = 0; i < map_outputs_to_pd_var.size(); ++i) {
      auto pd_var_desc = map_outputs_to_pd_var[i];
      auto symbol = var_to_symbol.at(output_keys.at(i));
      auto out = GetOrCreateDeviceTensor(symbol, pd_var_desc);
      auto* addr = (out->initialized()) ? (out->data()) : nullptr;
      dev_outputs_[graph_id].emplace_back(addr);
      dev_output_types_[graph_id].emplace_back(out->dtype());
      VLOG(6) << "alloc output Memory, symbol:" << symbol << ", addr:" << addr;
    }
    auto map_ref_out_to_weight = resources.map_ref_out_to_weight;
    auto map_inputs_to_pd_var = resources.map_inputs_to_pd_var;
    uint64_t exec_output_count = exec->NumOfOutputs();
    uint64_t exec_output_size[exec_output_count] = {0};
    exec->OutputSizeList(exec_output_size);
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
          platform::errors::NotFound(
              "the %zu output size[%llu] is not same with the %zu input size",
              i,
              output_size,
              input_idx,
              output_size));
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
      auto extra = GetOrCreateDeviceTensor(symbol, inputs_to_pd_var);
      auto* addr = (extra->initialized()) ? (extra->data()) : nullptr;
      dev_outputs_[graph_id].emplace_back(addr);
      dev_output_types_[graph_id].emplace_back(extra->dtype());
      VLOG(6) << "alloc extra output Memory, symbol:" << symbol
              << ", addr:" << addr;
    }
  }

  void GenerateGlobalSymbol() {
    const auto var_to_symbol = global_mem_ref_.var_to_symbol;
    const auto global_in_out_keys = global_mem_ref_.global_in_out_keys;
    // inputs
    const auto global_input_keys = global_in_out_keys.first;
    for (const auto key : global_input_keys) {
      auto symbol = var_to_symbol.at(key);
      global_input_symbols_.emplace_back(symbol);
      VLOG(6) << "GenerateGlobalSymbol global_input_symbols_, key:" << key
              << ", symbol:" << symbol;
    }
    // for performance
    global_tensor_skip_alloc_.insert(global_input_symbols_.begin(),
                                     global_input_symbols_.end());
    // outputs
    // const auto global_output_keys = global_in_out_keys.second;
    // for (const auto key : global_output_keys) {
    //   auto symbol = var_to_symbol.at(key);
    //   global_output_symbols_.emplace_back(symbol);
    //   VLOG(6) << "GenerateGlobalSymbol global_output_symbols_, key:" << key
    //           << ", symbol:" << symbol;
    // }

    // weight may not update
    if (is_train_) {
      const auto weights = global_mem_ref_.weights;
      const auto weight_to_symbol = global_mem_ref_.weight_to_symbol;
      const auto weight_update_params = global_mem_ref_.weight_update_params;
      for (const auto var_name : weights) {
        if (weight_update_params.count(var_name) == 0) {
          auto symbol = weight_to_symbol.at(var_name);
          non_trainable_weight_symbols_.emplace_back(
              std::make_pair(var_name, symbol));
          VLOG(6)
              << "GenerateGlobalSymbol non_trainable_weight_symbols_, var name:"
              << var_name << ", symbol:" << symbol;
        }
      }
    }
  }

  void GenerateGlobalMem() {
    const auto var_to_symbol = global_mem_ref_.var_to_symbol;
    const auto global_in_out_keys = global_mem_ref_.global_in_out_keys;

    // inputs
    const auto global_input_keys = global_in_out_keys.first;
    for (const auto key : global_input_keys) {
      auto symbol = var_to_symbol.at(key);
      size_t cnt = symbol_to_memory_.count(symbol);
      PADDLE_ENFORCE_NE(
          cnt,
          0,
          platform::errors::NotFound(
              "Device memory should be already allocated, key:%s, symbol:%s",
              key.c_str(),
              symbol.c_str()));
      global_inputs_.emplace_back(symbol_to_memory_[symbol]);
      VLOG(6) << "GenerateGlobalMem global_inputs_, key:" << key
              << ", symbol:" << symbol;
    }

    // outputs
    const auto global_output_keys = global_in_out_keys.second;
    for (const auto key : global_output_keys) {
      auto symbol = var_to_symbol.at(key);
      size_t cnt = symbol_to_memory_.count(symbol);
      PADDLE_ENFORCE_NE(
          cnt,
          0,
          platform::errors::NotFound(
              "Device memory should be already allocated, key:%s, symbol:%s",
              key.c_str(),
              symbol.c_str()));
      global_outputs_.emplace_back(symbol_to_memory_[symbol]);
      VLOG(6) << "GenerateGlobalMem global_outputs_, key:" << key
              << ", symbol:" << symbol
              << ", addr:" << symbol_to_memory_[symbol]->data();
    }

    // weights
    const auto weights = global_mem_ref_.weights;
    const auto weight_to_symbol = global_mem_ref_.weight_to_symbol;
    for (const auto var_name : weights) {
      auto symbol = weight_to_symbol.at(var_name);
      size_t cnt = symbol_to_memory_.count(symbol);
      PADDLE_ENFORCE_NE(
          cnt,
          0,
          platform::errors::NotFound("Device memory should be already "
                                     "allocated, var_name:%s, symbol:%s",
                                     var_name.c_str(),
                                     symbol.c_str()));
      global_weights_[var_name] = symbol_to_memory_[symbol];  // non-feed input
      VLOG(6) << "GenerateGlobalMem global_weights_, var_name:" << var_name
              << ", symbol:" << symbol
              << ", addr:" << symbol_to_memory_[symbol]->data();
    }

    // allreduce, if not enable tensor fusion
    if (is_distributed_ && !collective_tensor_fusion_) {
      const auto allreduce_params = global_mem_ref_.allreduce_params;
      for (const auto& param : allreduce_params) {
        auto input_symbol = param.in_out_desc[kCollectiveParamsInput].symbol;
        size_t cnt = symbol_to_memory_.count(input_symbol);
        PADDLE_ENFORCE_NE(
            cnt,
            0,
            platform::errors::NotFound("Device memory of allreduce input "
                                       "should be already allocated, symbol:%s",
                                       input_symbol.c_str()));
        auto output_symbol = param.in_out_desc[kCollectiveParamsOutput].symbol;
        cnt = symbol_to_memory_.count(output_symbol);
        PADDLE_ENFORCE_NE(
            cnt,
            0,
            platform::errors::NotFound("Device memory of allreduce output "
                                       "should be already allocated, symbol:%s",
                                       output_symbol.c_str()));
        allreduce_tensors_.emplace_back(std::make_pair(
            symbol_to_memory_[input_symbol], symbol_to_memory_[output_symbol]));
      }
    }
  }

  void AllocCollectiveTensorMem(
      const std::vector<CollectiveParams>& collective_params,
      const std::unordered_map<size_t, std::vector<int64_t>>& offsets,
      const std::unordered_map<size_t, int64_t>& total_size,
      size_t end_idx,
      bool reuse_input) {
    size_t start = end_idx - offsets.at(kCollectiveParamsInput).size();
    PADDLE_ENFORCE_EQ(
        ((end_idx >= offsets.at(kCollectiveParamsInput).size()) &&
         end_idx <= collective_params.size()),
        true,
        platform::errors::OutOfRange(
            "index out of range, start:%zu, end:%zu, offsets size:%zu, "
            "collective_params size:%zu",
            start,
            end_idx,
            offsets.at(kCollectiveParamsInput).size(),
            collective_params.size()));
    VLOG(6) << "AllocCollectiveTensorMem start:" << start << ", end:" << end_idx
            << ", offsets size:" << offsets.at(kCollectiveParamsInput).size()
            << ", collective_params size:" << collective_params.size();

    auto alloc_tensor = [&](size_t param_idx,
                            const PaddleVarDesc& var_desc) -> TensorPtr {
      auto tensor = CreateTensor(place_, var_desc);
      VLOG(6) << "AllocCollectiveTensorMem alloc "
              << ((param_idx == static_cast<size_t>(kCollectiveParamsInput))
                      ? "input"
                      : "output")
              << " size:" << total_size.at(param_idx)
              << ", addr:" << tensor->data();
      for (size_t i = 0; i < offsets.at(param_idx).size(); ++i) {
        auto var_desc =
            collective_params[start + i].in_out_desc[param_idx].var_desc;
        auto symbol =
            collective_params[start + i].in_out_desc[param_idx].symbol;
        auto sub_tensor =
            CreateTensor(tensor, var_desc, offsets.at(param_idx)[i]);
        this->symbol_to_memory_.emplace(symbol, sub_tensor);
        this->memory_to_sub_memory_[tensor].emplace_back(sub_tensor);
        VLOG(6) << "AllocCollectiveTensorMem alloc sub tensor offset:"
                << offsets.at(param_idx)[i] << ", symbol:" << symbol
                << ", addr:" << sub_tensor->data();
      }
      this->memory_stores_[kMemoryTypeCollective].emplace_back(tensor);
      return tensor;
    };

    PaddleVarType data_type = collective_params[start]
                                  .in_out_desc[kCollectiveParamsInput]
                                  .var_desc.data_type;
    int64_t input_numel = total_size.at(kCollectiveParamsInput) /
                          framework::SizeOfType(data_type);
    int64_t output_numel = total_size.at(kCollectiveParamsOutput) /
                           framework::SizeOfType(data_type);
    // input
    auto input_desc = PaddleVarDesc("collective_fusion_input",
                                    {1, input_numel},
                                    data_type,
                                    total_size.at(kCollectiveParamsInput));
    TensorPtr input_tensor = alloc_tensor(kCollectiveParamsInput, input_desc);
    TensorPtr output_tensor = input_tensor;
    if (!reuse_input) {
      auto output_desc = PaddleVarDesc("collective_fusion_output",
                                       {1, input_numel},
                                       data_type,
                                       total_size.at(kCollectiveParamsInput));
      output_tensor = alloc_tensor(kCollectiveParamsOutput, output_desc);
    }
    allreduce_tensors_.emplace_back(
        std::make_pair(input_tensor, output_tensor));
    VLOG(6) << "AllocCollectiveTensorMem reuse_input:" << reuse_input
            << ", input_numel:" << input_numel
            << ", output_numel:" << output_numel;
  }

  void CollectiveTensorFusion(
      const std::vector<CollectiveParams>& collective_params,
      int64_t align_size = kMemAlignSize) {
    if (collective_params.empty()) {
      return;
    }
    VLOG(3) << "Enable CollectiveTensorFusion with MaxFusionMemSize:"
            << kMaxFusionMemSize;
    bool reuse_input = collective_params[0].reuse_input;
    PaddleVarType data_type = collective_params[0]
                                  .in_out_desc[kCollectiveParamsInput]
                                  .var_desc.data_type;
    int64_t input_offset = 0;
    int64_t output_offset = 0;
    std::unordered_map<size_t, std::vector<int64_t>> offsets;
    std::unordered_map<size_t, int64_t> total_sizes;
    for (size_t i = 0; i < collective_params.size(); ++i) {
      const auto& input_var_desc =
          collective_params[i].in_out_desc[kCollectiveParamsInput].var_desc;
      const auto& output_var_desc =
          collective_params[i].in_out_desc[kCollectiveParamsOutput].var_desc;
      PADDLE_ENFORCE_EQ(
          (input_var_desc.data_type == output_var_desc.data_type),
          true,
          platform::errors::Unavailable(
              "data type should be same, data_type:%d, input data_type:%d, "
              "output data_type:%d",
              data_type,
              input_var_desc.data_type,
              output_var_desc.data_type));
      int64_t input_size = align_size > 0
                               ? ((input_var_desc.data_size + align_size - 1) /
                                  align_size * align_size)
                               : input_var_desc.data_size;  // align if need
      int64_t output_size =
          align_size > 0 ? ((output_var_desc.data_size + align_size - 1) /
                            align_size * align_size)
                         : output_var_desc.data_size;  // align if need
      int64_t max_size =
          std::max(input_offset + input_size, output_offset + output_size);
      bool same_dtype = data_type == input_var_desc.data_type;
      if (max_size > kMaxFusionMemSize || !same_dtype) {
        auto tensor_max_size = std::max(input_size, output_size);
        PADDLE_ENFORCE_NE(
            tensor_max_size > kMaxFusionMemSize,
            true,
            platform::errors::PreconditionNotMet(
                "tensor size %lu is greater than MAX size:%lu, var name:%s",
                tensor_max_size,
                kMaxFusionMemSize,
                input_var_desc.var_name.c_str()));
        VLOG(3) << "AllocCollectiveTensorMem when "
                << (same_dtype ? "threshold exceeded"
                               : "data type is different");
        AllocCollectiveTensorMem(
            collective_params, offsets, total_sizes, i, reuse_input);
        input_offset = 0;
        output_offset = 0;
        offsets.clear();
        total_sizes.clear();
        data_type = collective_params[i]
                        .in_out_desc[kCollectiveParamsInput]
                        .var_desc.data_type;
      }
      offsets[kCollectiveParamsInput].emplace_back(input_offset);
      offsets[kCollectiveParamsOutput].emplace_back(output_offset);
      input_offset += input_size;
      output_offset += output_size;
      total_sizes[kCollectiveParamsInput] = input_offset;
      total_sizes[kCollectiveParamsOutput] = output_offset;
    }
    if (!offsets[kCollectiveParamsInput].empty()) {
      VLOG(3) << "AllocCollectiveTensorMem last";
      AllocCollectiveTensorMem(collective_params,
                               offsets,
                               total_sizes,
                               collective_params.size(),
                               reuse_input);
    }
  }

  void ReplaceWeightTensorsInScope(
      const std::unordered_map<std::string, TensorPtr>& var_to_device_weight) {
    for (const auto var_to_dev_weight : var_to_device_weight) {
      auto var = scope_->FindVar(var_to_dev_weight.first);
      if (var != nullptr) {
        auto var_tensor = var->GetMutable<Tensor>();
        auto dev_weight = var_to_dev_weight.second;
        if (var_tensor->initialized()) {
          VLOG(6) << "Copy weight:" << var_to_dev_weight.first << " from "
                  << var_tensor->place() << " to " << dev_weight->place();
          paddle::framework::TensorCopySync(
              *var_tensor, place_, dev_weight.get());
          var_tensor->ShareDataWith(*dev_weight);
        } else {
          var_tensor->ShareDataWith(*dev_weight);
          VLOG(6) << "Init weight to " << var_tensor->place()
                  << ", addr:" << var_tensor->data();
        }
        auto src_tensor = var->GetMutable<Tensor>();
        VLOG(6) << "After Copy or Init weight, " << var_to_dev_weight.first
                << " locates on:" << src_tensor->place()
                << ", addr:" << src_tensor->data();
      }
    }
  }

  void InitNonTrainableWeights() {
    std::unordered_map<std::string, TensorPtr> non_trainable_weights;
    for (const auto& var_symbol : non_trainable_weight_symbols_) {
      auto var_name = var_symbol.first;
      auto symbol = var_symbol.second;
      size_t cnt = symbol_to_memory_.count(symbol);
      PADDLE_ENFORCE_NE(
          cnt,
          0,
          platform::errors::NotFound("Non-trainable weight should be already "
                                     "allocated, var:%s, symbol:%s",
                                     var_name.c_str(),
                                     symbol.c_str()));
      non_trainable_weights[var_name] = symbol_to_memory_[symbol];
    }
    ReplaceWeightTensorsInScope(non_trainable_weights);
  }

  void AllocWeightMemory(const std::vector<WeightUpdateParams>& params,
                         const std::vector<int64_t>& offsets,
                         int64_t total_size,
                         bool is_trans_weights) {
    const auto weight_to_symbol = global_mem_ref_.weight_to_symbol;
    PADDLE_ENFORCE_EQ(
        params.size(),
        offsets.size(),
        platform::errors::Unavailable(
            "params size(%zu) should be same to offsets size(%zu)",
            params.size(),
            offsets.size()));

    auto tmp_data_type = paddle::framework::proto::VarType::FP32;
    auto weight_tensor_desc = PaddleVarDesc(
        "weights_fusion",
        {1,
         total_size /
             static_cast<int64_t>(framework::SizeOfType(tmp_data_type))},
        tmp_data_type,
        total_size);
    auto weights_tensor = CreateTensor(place_, weight_tensor_desc);
    auto weights_src_tensor = CreateTensor(place_, weight_tensor_desc);

    if (is_trans_weights) {
      memory_stores_[kMemoryTypeTransWeights].emplace_back(weights_tensor);
      memory_stores_[kMemoryTypeTransWeights].emplace_back(weights_src_tensor);
    } else {
      memory_stores_[kMemoryTypeWeights].emplace_back(weights_tensor);
      memory_stores_[kMemoryTypeWeights].emplace_back(weights_src_tensor);
    }

    std::unordered_map<std::string, TensorPtr> var_to_device_weight;

    for (size_t i = 0; i < params.size(); ++i) {
      auto var_desc = params[i].var_desc;
      auto var_name = var_desc.var_name;
      size_t cnt = weight_to_symbol.count(var_name);
      PADDLE_ENFORCE_NE(cnt,
                        0,
                        platform::errors::NotFound(
                            "var should have symbol in weight_to_symbol, "
                            "var_name:%s",
                            var_name.c_str()));
      auto weight_symbol = weight_to_symbol.at(var_name);
      auto src_symbol = params[i].symbol;
      // weights
      auto sub_weight_tensor =
          CreateTensor(weights_tensor, var_desc, offsets[i]);
      symbol_to_memory_[weight_symbol] = sub_weight_tensor;
      memory_to_sub_memory_[weights_tensor].emplace_back(sub_weight_tensor);
      global_weights_[var_name] = sub_weight_tensor;
      var_to_device_weight[var_name] = sub_weight_tensor;
      // extra output
      auto sub_src_tensor =
          CreateTensor(weights_src_tensor, var_desc, offsets[i]);
      symbol_to_memory_[src_symbol] = sub_src_tensor;
      memory_to_sub_memory_[weights_src_tensor].emplace_back(sub_src_tensor);

      VLOG(6) << "AllocWeightMemory alloc sub tensor offset:" << offsets[i]
              << ", size:" << var_desc.data_size
              << ", weight_symbol:" << weight_symbol
              << ", addr:" << sub_weight_tensor->data()
              << ", src_symbol:" << src_symbol
              << ", addr:" << sub_src_tensor->data();
    }
    std::string fused_name =
        is_trans_weights ? ("fused_trans_weights") : ("fused_weights");
    weights_update_.emplace(fused_name,
                            std::make_pair(weights_tensor, weights_src_tensor));

    ReplaceWeightTensorsInScope(var_to_device_weight);
    total_weights_.emplace_back(weights_tensor);

    VLOG(6) << "AllocWeightMemory fused_name:" << fused_name
            << ", weight size:" << params.size()
            << ", total data size:" << total_size
            << ", weights_tensor:" << weights_tensor->data()
            << ", weights_src_tensor:" << weights_src_tensor->data();
  }

  void WeightTensorFusion(int64_t align_size, bool is_trans_weights) {
    const auto weight_update_params = global_mem_ref_.weight_update_params;
    const auto global_transed_weights_info = global_mem_ref_.weights_trans_info;
    std::map<std::string, WeightUpdateParams> update_params;
    if (is_trans_weights) {
      for (const auto& name_to_trans_info : global_transed_weights_info) {
        auto var_name = name_to_trans_info.first;
        if (weight_update_params.count(var_name) == 0) {
          continue;
        }
        update_params.emplace(
            std::make_pair(var_name, weight_update_params.at(var_name)));
      }
    } else {
      for (const auto& update_param : weight_update_params) {
        auto var_name = update_param.first;
        if (global_transed_weights_info.count(var_name) > 0) {
          continue;
        }
        update_params.emplace(
            std::make_pair(var_name, weight_update_params.at(var_name)));
      }
    }
    int64_t offset = 0;
    std::vector<int64_t> offsets;
    std::vector<WeightUpdateParams> params;
    int64_t total_size = 0;
    for (const auto& update_param : update_params) {
      auto var_name = update_param.first;
      auto param = update_param.second;
      auto var_desc = param.var_desc;
      int64_t size = align_size > 0 ? ((var_desc.data_size + align_size - 1) /
                                       align_size * align_size)
                                    : var_desc.data_size;  // align if need
      if (size + offset > kMaxFusionMemSize) {
        VLOG(3) << "AllocWeightMemory when threshold exceeded";
        AllocWeightMemory(params, offsets, total_size, is_trans_weights);
        offset = 0;
        total_size = 0;
        offsets.clear();
        params.clear();
      }
      offsets.emplace_back(offset);
      params.emplace_back(param);
      offset += size;
      total_size = offset;
    }
    if (!offsets.empty()) {
      VLOG(3) << "AllocWeightMemory last";
      AllocWeightMemory(params, offsets, total_size, is_trans_weights);
    }
  }

  void WeightTensorFusion(int64_t align_size = kMemAlignSize) {
    WeightTensorFusion(align_size, true);
    WeightTensorFusion(align_size, false);
  }

  void GetWeightsFromScope() {
    const auto weights = global_mem_ref_.weights;
    const auto weight_to_symbol = global_mem_ref_.weight_to_symbol;
    for (const auto var_name : weights) {
      auto var = scope_->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          var, "Failed to find %s in scope.", var_name.c_str());
      auto var_tensor = var->GetMutable<Tensor>();
      if (var_tensor->numel() <= 0) {
        continue;
      }
      PADDLE_ENFORCE_EQ(
          var_tensor->initialized(),
          true,
          platform::errors::PreconditionNotMet(
              "Weight in scope should already been initialized, name:%s",
              var_name.c_str()));
      auto symbol = weight_to_symbol.at(var_name);
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

  void TransWeightsToHost() {
    auto dst_gcu_tensor = memory_stores_[kMemoryTypeTransWeightsDst].at(0);
    auto dst_cpu_tensor = memory_stores_[kMemoryTypeTransWeightsDst].at(1);
    paddle::framework::TensorCopySync(
        *dst_gcu_tensor, platform::CPUPlace(), dst_cpu_tensor.get());
  }

  void TransWeightsToDevice() {
    auto src_gcu_tensor = memory_stores_[kMemoryTypeTransWeightsSrc].at(0);
    auto src_cpu_tensor = memory_stores_[kMemoryTypeTransWeightsSrc].at(1);
    paddle::framework::TensorCopySync(
        *src_cpu_tensor, place_, src_gcu_tensor.get());
  }

  void IteratorSync() { stream_->Synchronize(); }

  void Synchronize() {
    if (!is_train_) {
      VLOG(0) << "Graph is not trainable, will not synchronize";
      return;
    }
    // std::vector<GcuTransInfo> trans_infos;
    std::vector<std::future<void>> vector_future;
    VLOG(0) << "GcuExecutor start to Synchronize";
    Recorder recorder("true");
    auto begin = system_clock::now();

    if (!no_need_trans_weights_) {
      InverseTransWeightsInHost(vector_future);
      std::for_each(vector_future.begin(),
                    vector_future.end(),
                    [](const std::future<void>& f) { f.wait(); });
      TransWeightsToDevice();
    }

    auto sync_cost = recorder.Cost(begin, system_clock::now());
    VLOG(0) << "GcuExecutor Synchronize in " << sync_cost << " ms.";
  }

  explicit GcuExecutorImpl(const framework::Scope* scope) {
    scope_ = scope;
    int device_id = runtime::GcuGetCurrentDevice();
    place_ = CustomPlace("gcu", device_id);
    ctx_ = runtime::GcuGetContext(device_id);
    PADDLE_ENFORCE_NE(
        ctx_, nullptr, platform::errors::NotFound("create runtime ctx failed"));
    stream_ = ctx_->default_exe_stream;
    PADDLE_ENFORCE_NE(
        stream_, nullptr, platform::errors::NotFound("create stream failed"));
    dma_stream_ = ctx_->default_dma_stream;
    PADDLE_ENFORCE_NE(dma_stream_,
                      nullptr,
                      platform::errors::NotFound("create stream failed"));

    auto rt_info = runtime::GcuGetRuntimeInfo(device_id);
    is_distributed_ = rt_info->is_distributed;
    if (is_distributed_) {
      PADDLE_ENFORCE_NE(
          distributed::ProcessGroupIdMap::GetInstance().count(kGlobalGroupID),
          0,
          platform::errors::NotFound("Process group should be already inited"));
      process_group_ =
          distributed::ProcessGroupIdMap::GetInstance().at(kGlobalGroupID);
    }
    VLOG(1) << "Init GcuExecutorImpl for device_id:" << device_id
            << ", place:" << place_ << ", is_distributed:" << is_distributed_;
  }

  void ReleaseResource() {
    ReleaseMemory();
    executables_.clear();
    TransformUtil::GraphToGcuExecutable(program_key_, {}, {});
  }

  void ReleaseMemory() {
    // GCU memory is managed uniformly in symbol_to_memory_
    global_inputs_.clear();
    global_outputs_.clear();
    global_weights_.clear();
    weights_update_.clear();
    total_weights_.clear();
    allreduce_tensors_.clear();
    symbol_to_memory_.clear();
    var_to_transed_info_.clear();
    memory_to_sub_memory_.clear();
    memory_stores_.clear();
  }

  void ResetScope(const framework::Scope* scope) { scope_ = scope; }

  GcuExecutorImpl(const GcuExecutorImpl& impl) = default;

  ~GcuExecutorImpl() {}

  GcuExecutorImpl& operator=(const GcuExecutorImpl& impl) = default;

 private:
  std::once_flag init_once_;
  GcuCtxPtr ctx_ = nullptr;
  std::string program_key_;
  GcuStreamPtr stream_ = nullptr;
  GcuStreamPtr dma_stream_ = nullptr;
  CustomPlace place_;
  const framework::Scope* scope_ = nullptr;
  bool weight_init_ = false;
  std::vector<ExecutablePtr> executables_;
  GlobalMemoryRef global_mem_ref_;
  TensorPtr tmp_input_tensor_ = nullptr;
  TensorPtr tmp_output_tensor_ = nullptr;
  std::unordered_map<size_t, std::vector<void*>> dev_inputs_;
  std::unordered_map<size_t, std::vector<void*>> dev_outputs_;
  std::unordered_map<size_t, std::vector<DataType>> dev_input_types_;
  std::unordered_map<size_t, std::vector<DataType>> dev_output_types_;
  std::unordered_map<std::string, TensorPtr> symbol_to_memory_;
  std::unordered_map<std::string, GcuTransInfo> var_to_transed_info_;
  std::unordered_map<std::string, std::vector<TensorPtr>> memory_stores_;
  std::unordered_map<TensorPtr, std::vector<TensorPtr>> memory_to_sub_memory_;
  std::vector<TensorPtr> global_inputs_;
  std::vector<TensorPtr> global_outputs_;
  std::vector<std::string> global_input_symbols_;
  //   std::vector<std::string> global_output_symbols_;
  std::unordered_set<std::string> global_tensor_skip_alloc_;
  std::unordered_map<std::string, std::vector<std::pair<size_t, size_t>>>
      global_input_to_dev_input_;
  std::map<std::string, TensorPtr> global_weights_;  // broadcast in order
  std::vector<TensorPtr> total_weights_;
  std::vector<std::pair<std::string, std::string>>
      non_trainable_weight_symbols_;
  // dst_addr, src_addr, data_size
  std::unordered_map<std::string, std::pair<TensorPtr, TensorPtr>>
      weights_update_;
  std::vector<std::pair<TensorPtr, TensorPtr>> allreduce_tensors_;
  bool is_distributed_ = false;
  bool is_train_ = true;
  bool leaf_output_ = false;
  std::string running_mode_ = RunningMode::SERIAL;
  ProcessGroupPtr process_group_ = nullptr;
  bool collective_tensor_fusion_ = true;
  bool weight_sync_smoothly_ = true;
  int64_t sync_interval_ = 1;
  int64_t train_iters_ = 0;
  bool no_need_trans_weights_ = false;
};

GcuExecutor::GcuExecutor(const framework::Scope* scope) {
  impl_ = std::make_shared<GcuExecutorImpl>(scope);
}

void GcuExecutor::ReleaseResource() { impl_->ReleaseResource(); }

void GcuExecutor::ReleaseMemory() { impl_->ReleaseMemory(); }

void GcuExecutor::ResetScope(const framework::Scope* scope) {
  impl_->ResetScope(scope);
}

void GcuExecutor::RunGcuOp(const std::vector<const Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs,
                           const paddle::framework::ExecutionContext& ctx,
                           const std::string& program_key,
                           const int train_flag,
                           const framework::Scope* curr_scope) {
  impl_->RunGcuOp(inputs, outputs, ctx, program_key, train_flag, curr_scope);
}

void GcuExecutor::Synchronize() { impl_->Synchronize(); }

void Synchronize(const framework::ProgramDesc& program) {
  auto& block = program.Block(framework::kRootBlockIndex);
  std::string program_key;
  for (auto& op_desc : block.AllOps()) {
    if (op_desc->Type() == "gcu_runtime") {
      auto attr = op_desc->GetAttr(kGcuProgramKey);
      program_key = PADDLE_GET_CONST(std::string, attr);
      break;
    }
  }
  auto manager = GcuExecutorManager::GetInstance();
  std::shared_ptr<GcuExecutor> gcu_exec = manager->Find(program_key);
  PADDLE_ENFORCE_NOT_NULL(gcu_exec, "Failed to find gcu_exec.");
  gcu_exec->Synchronize();
}
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
