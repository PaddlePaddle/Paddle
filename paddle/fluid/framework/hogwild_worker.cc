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

#include <array>
#include <ctime>

#include "paddle/fluid/framework/barrier.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/new_executor/interpreter/dependency_builder.h"
#include "paddle/fluid/operators/controlflow/conditional_block_op_helper.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/lodtensor_printer.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#if defined PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif
#include "paddle/phi/core/flags.h"
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_GPU_GRAPH)
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#endif
#include "paddle/fluid/framework/program_utils.h"

PHI_DECLARE_bool(enable_exit_when_partial_worker);
PHI_DECLARE_int32(enable_adjust_op_order);
PHI_DEFINE_EXPORTED_bool(
    gpugraph_force_device_batch_num_equal,
    false,
    "enable force_device_batch_num_equal, default false");
PHI_DECLARE_bool(enable_dump_main_program);
namespace paddle {
namespace framework {

std::atomic<bool> HogwildWorker::quit_flag_(false);
Barrier g_barrier;

void HogwildWorker::Initialize(const TrainerDesc &desc) {
  fetch_config_ = desc.fetch_config();
  param_ = desc.hogwild_param();
  skip_ops_.resize(param_.skip_ops_size());
  for (int i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }
  use_cvm_ = desc.use_cvm();
  thread_barrier_ = desc.thread_barrier();

  for (int i = 0; i < param_.stat_var_names_size(); ++i) {
    std::string name = param_.stat_var_names(i);
    stat_var_name_map_[name] = 1;
    skip_vars_.push_back(name);
  }
}
int HogwildWorker::IsParameter(const std::string &name, bool full_match) {
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_GPU_GRAPH)
  auto gpu_ps = PSGPUWrapper::GetInstance();
  bool last_device = ((thread_num_ - 1) == thread_id_);
#endif
  if (full_match) {
    auto it = params2rootid_.find(name);
    if (it == params2rootid_.end()) {
      return -1;
    }
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_GPU_GRAPH)
    if (last_device && !gpu_ps->IsKeyForSelfRank(it->second)) {
      free_param_vars_.insert(name);
    }
#endif
    if (it->second == nccl_rank_id_) {
      return 1;
    }
    return 0;
  } else {
    // moment, acc
    for (auto it = params2rootid_.begin(); it != params2rootid_.end(); ++it) {
      if (strncmp(name.c_str(), it->first.c_str(), it->first.length()) != 0) {
        continue;
      }
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_GPU_GRAPH)
      if (last_device && !gpu_ps->IsKeyForSelfRank(it->second)) {
        free_param_vars_.insert(name);
      }
#endif
      if (it->second == nccl_rank_id_) {
        return 1;
      }
      return 0;
    }
    return -1;
  }
}
void HogwildWorker::BuildShardingDepends(const ProgramDesc &program) {
  nccl_rank_id_ = place_.GetDeviceId();
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_GPU_GRAPH)
  auto gpu_ps = PSGPUWrapper::GetInstance();
  nccl_rank_id_ = gpu_ps->GetNCCLRankId(nccl_rank_id_);
#endif

  auto &block = program.Block(0);
  auto all_desc = block.AllOps();

  for (auto &op_desc : all_desc) {
    // broadcast op
    if (op_desc->Type() != "c_broadcast") {
      continue;
    }
    int root_id = op_desc->GetAttrIfExists<int>("root");
    int ring_id = op_desc->GetAttrIfExists<int>("ring_id");
    if (ring_id >= 0 && ring_id != ring_id_) {
      ring_id_ = ring_id;
    }
    std::string new_name;
    for (auto &o : op_desc->Inputs()) {
      for (auto &name : o.second) {
        // amp
        size_t pos = name.find(".cast_fp16");
        if (pos != std::string::npos) {
          new_name = name.substr(0, pos);
        } else {
          new_name = name;
        }
        auto var = block.FindVar(new_name);
        if (!var->Persistable() || !var->IsParameter()) {
          continue;
        }
        if (params2rootid_.find(new_name) != params2rootid_.end()) {
          continue;
        }
        params2rootid_.insert(std::make_pair(new_name, root_id));
      }
    }
  }
  if (params2rootid_.empty()) {
    return;
  }
  sharding_mode_ = true;
  // check find
  for (auto &var : block.AllVars()) {
    if (!var->Persistable()) {
      continue;
    }
    int ret = IsParameter(var->Name(), var->IsParameter());
    if (ret < 0 || ret == 1) {
      if (ret == 1) {
        persist_param_vars_.insert(var->Name());
      }
      continue;
    }
    if (var->IsParameter()) {
      unpersist_vars_.insert(var->Name());
    } else {
      remove_vars_.insert(var->Name());
    }
  }
  for (auto &op_desc : all_desc) {
    bool find = false;
    for (auto &o : op_desc->Inputs()) {
      for (auto &name : o.second) {
        if (remove_vars_.find(name) == remove_vars_.end()) {
          continue;
        }
        find = true;
        break;
      }
      if (find) {
        break;
      }
    }
    if (find) {
      remove_ops_.insert(op_desc);
    }
  }

  // reset dump param
  if (need_dump_param_ && dump_param_ != nullptr) {
    for (auto &name : *dump_param_) {
      auto var = block.FindVar(name);
      if (var == nullptr) {
        continue;
      }
      std::string new_name = name;
      size_t pos = new_name.find("@");
      if (pos > 0) {
        new_name = name.substr(0, pos);
      }
      if (persist_param_vars_.find(new_name) == persist_param_vars_.end()) {
        continue;
      }
      shard_dump_params_.push_back(name);
    }
    dump_param_ = &shard_dump_params_;
  }
  // reset dump fields
  if (need_dump_field_ && dump_fields_ != nullptr) {
    for (auto &name : *dump_fields_) {
      auto var = block.FindVar(name);
      if (var == nullptr) {
        continue;
      }
      if (remove_vars_.find(name) != remove_vars_.end()) {
        continue;
      }
      shard_dump_fields_.push_back(name);
    }
    dump_fields_ = &shard_dump_fields_;
  }
  // debug proto
  if (FLAGS_enable_dump_main_program) {
    ProgramDesc desc(program);
    auto new_block = desc.MutableBlock(0);
    for (auto &name : remove_vars_) {
      new_block->RemoveVar(name);
    }
    for (auto &name : unpersist_vars_) {
      auto var = new_block->FindVar(name);
      var->SetPersistable(false);
      var->SetIsParameter(false);
    }
    std::vector<OpDesc *> remove_ops;
    for (auto &op_desc : new_block->AllOps()) {
      bool find = false;
      for (auto &o : op_desc->Inputs()) {
        for (auto &name : o.second) {
          if (remove_vars_.find(name) == remove_vars_.end()) {
            continue;
          }
          find = true;
          break;
        }
        if (find) {
          break;
        }
      }
      if (find) {
        remove_ops.push_back(op_desc);
      }
    }
    for (auto &op : remove_ops) {
      new_block->RemoveOpInternal(op);
    }
    desc.Flush();
    char name[512];
    snprintf(name, sizeof(name), "thread_program_%d", nccl_rank_id_);
    DumpProgramDescFile(name, desc);
  }

  VLOG(0) << "device id=" << int(place_.GetDeviceId())
          << ", nccl rank=" << nccl_rank_id_
          << ", total param count=" << params2rootid_.size()
          << ", remove op count=" << remove_ops_.size()
          << ", remove var count=" << remove_vars_.size()
          << ", unpersist var count=" << unpersist_vars_.size()
          << ", dump param count=" << shard_dump_params_.size()
          << ", dump fields count=" << shard_dump_fields_.size();
}
void HogwildWorker::CreateThreadOperators(const ProgramDesc &program) {
  auto &block = program.Block(0);
  op_names_.clear();
  auto all_desc = block.AllOps();
  std::set<size_t> remove_ids;
  size_t op_index = 0;
  for (auto &op_desc : all_desc) {
    // skip feed fetch op
    if (op_desc->Type() == "feed" || op_desc->Type() == "fetch") {
      for (auto &o : op_desc->Inputs()) {
        skip_vars_.insert(skip_vars_.end(), o.second.begin(), o.second.end());
      }
      for (auto &o : op_desc->Outputs()) {
        skip_vars_.insert(skip_vars_.end(), o.second.begin(), o.second.end());
      }
    }
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op_desc->Type().find(skip_ops_[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (need_skip) {
      continue;
    }
    // skip remove ops
    if (remove_ops_.find(op_desc) != remove_ops_.end()) {
      if (FLAGS_enable_adjust_op_order) {
        remove_ids.insert(op_index);
      } else {
        continue;
      }
    }

    op_names_.push_back(op_desc->Type());
    ops_.emplace_back(OpRegistry::CreateOp(*op_desc));
    op_index++;
  }
  if (FLAGS_enable_adjust_op_order) {
    std::vector<size_t> new_order;
    size_t start_index = 0;
    for (auto &op : ops_) {
      int op_role = op->Attr<int>("op_role");
      if ((op_role == static_cast<int>(OpRole::kForward)) ||
          (op_role == (static_cast<int>(OpRole::kForward) |
                       static_cast<int>(OpRole::kLoss))) ||
          (op_role == static_cast<int>(OpRole::kLRSched))) {
        start_index++;
      } else {
        break;
      }
    }

    if (start_index < ops_.size()) {
      interpreter::DependencyBuilderSimplify depend_builder;
      // depend_builder.Build(ops_, start_index, sharding_mode_);  hbm not safe
      // shoud run in debug model need to fix
      depend_builder.Build(ops_, start_index, false);
      new_order = depend_builder.get_new_exexutor_order();
      std::vector<std::unique_ptr<OperatorBase>> new_ops;
      std::vector<size_t> final_order;
      for (auto index : new_order) {
        if (remove_ids.count(index) == 0) {
          new_ops.push_back(std::move(ops_[index]));
          final_order.push_back(index);
        }
      }
      new_order = final_order;
      ops_ = std::move(new_ops);
    }
  }
  operators::PrepareSafeEagerDeletionOnConditionalOpAndConditionalGradOp(
      program, 0, ops_);
  // not need gc
  int64_t max_memory_size = GetEagerDeletionThreshold();
  if (max_memory_size < 0) {
    return;
  }
  // skip dump fields
  if (need_dump_field_ && dump_fields_ != nullptr) {
    skip_vars_.insert(
        skip_vars_.end(), dump_fields_->begin(), dump_fields_->end());
  }
  // skip dump params
  if (need_dump_param_ && dump_param_ != nullptr) {
    skip_vars_.insert(
        skip_vars_.end(), dump_param_->begin(), dump_param_->end());
  }
  int fetch_var_num = fetch_config_.fetch_var_names_size();
  if (fetch_var_num > 0) {
    for (int i = 0; i < fetch_var_num; ++i) {
      std::string name = fetch_config_.fetch_var_names(i);
      skip_vars_.push_back(name);
    }
  }
  unused_vars_ =
      GetUnusedVars(block, ops_, skip_vars_, &unpersist_vars_, sharding_mode_);
  // debug
  VLOG(1) << "device id=" << thread_id_ << "total op count=" << all_desc.size()
          << ", create op count=" << ops_.size()
          << ", skip vars count=" << skip_vars_.size()
          << ", unused vars op count=" << unused_vars_.size();
}
inline void PrintTensor(const std::string &name,
                        const std::string &info,
                        Scope *scope) {
  std::stringstream ss;
  platform::PrintVar(scope, name, info, &ss);
  std::cout << ss.str() << std::endl;
}
void HogwildWorker::CreateThreadScope(const ProgramDesc &program) {
  auto &block = program.Block(0);

  PADDLE_ENFORCE_NOT_NULL(
      root_scope_,
      platform::errors::NotFound(
          "Root scope should be set before creating thread scope."));

  thread_scope_ = &root_scope_->NewScope();

  int persist_total = 0;
  int persist_param = 0;
  int persist_share = 0;
  int persist_reset = 0;
  std::vector<std::string> del_var_names;
  for (auto &var : block.AllVars()) {
    auto name = var->Name();
    if (remove_vars_.find(name) != remove_vars_.end()) {
      if (free_param_vars_.find(name) != free_param_vars_.end()) {
        del_var_names.push_back(name);
        VLOG(1) << "remove need delete var name=" << name;
      }
      continue;
    }
    all_param_.push_back(name);
    if (var->Persistable()) {
      ++persist_total;
      if (stat_var_name_map_.find(name) != stat_var_name_map_.end()) {
        Variable *root_var = root_scope_->FindVar(name);
        CHECK(root_var != nullptr);
        auto root_tensor = root_var->Get<phi::DenseTensor>();
        if (root_tensor.place() == place_) {
          continue;
        }
        auto *ptr1 = thread_scope_->Var(name);
        InitializeVariable(ptr1, var->GetType());
        phi::DenseTensor *thread_tensor = ptr1->GetMutable<phi::DenseTensor>();
#define MemsetCallback(cpp_type, proto_type)                                 \
  do {                                                                       \
    if (framework::TransToProtoVarType(root_tensor.dtype()) == proto_type) { \
      SetZero<cpp_type>(thread_tensor, root_tensor);                         \
    }                                                                        \
  } while (0)
        _ForEachDataType_(MemsetCallback);
      }
#ifdef PADDLE_WITH_GPU_GRAPH
      else if (unpersist_vars_.find(name) == unpersist_vars_.end()) {
        Variable *root_var = root_scope_->FindVar(name);
        if (!root_var) {
          VLOG(0) << "not found var name=" << name;
          continue;
        }
        if (root_var->IsType<phi::SelectedRows>()) {
          continue;
        }
        ++persist_param;
        phi::DenseTensor *root_tensor =
            root_var->GetMutable<phi::DenseTensor>();
        if (place_ == root_tensor->place()) {
          ++persist_share;
          continue;
        }
        // reset tensor holder
        if (persist_param_vars_.find(name) != persist_param_vars_.end() &&
            platform::is_gpu_place(root_tensor->place())) {
          phi::DenseTensor cpu_tensor;
          TensorCopy(*root_tensor, platform::CPUPlace(), &cpu_tensor);
          root_tensor->MoveMemoryHolder();
          TensorCopy(cpu_tensor, place_, root_tensor);
          ++persist_reset;
        } else {
          auto *ptr = thread_scope_->Var(name);
          CHECK(proto::VarType::LOD_TENSOR == var->GetType());
          InitializeVariable(ptr, var->GetType());
          phi::DenseTensor *thread_tensor = ptr->GetMutable<phi::DenseTensor>();
          TensorCopy(*root_tensor, place_, thread_tensor);
          need_copy_vars_.push_back(name);
        }
      } else {
        if (free_param_vars_.find(name) != free_param_vars_.end()) {
          del_var_names.push_back(name);
          VLOG(0) << "unpersist need delete var name=" << name;
        }
        // sharding vars
        auto *ptr = thread_scope_->Var(name);
        InitializeVariable(ptr, var->GetType());
        // set dims
        auto dims = phi::make_ddim(var->GetShape());
        ptr->GetMutable<phi::DenseTensor>()->Resize(dims);
      }
#endif
    } else {
      auto *ptr = thread_scope_->Var(name);
      InitializeVariable(ptr, var->GetType());
    }
  }
  // multi node delete unused vars
  if (!del_var_names.empty()) {
    root_scope_->EraseVars(del_var_names);
  }
  VLOG(0) << "device id=" << thread_id_
          << ", total param count=" << all_param_.size()
          << ", persist count=" << persist_total << ", param=" << persist_param
          << ", share=" << persist_share << ", reset=" << persist_reset
          << ", need copy param count=" << need_copy_vars_.size()
          << ", delete vars count=" << del_var_names.size();
}
void HogwildWorker::Finalize() {
#ifdef PADDLE_WITH_HETERPS
  if (!sharding_mode_ && thread_id_ != 0) {
    return;
  }
  for (auto &name : need_copy_vars_) {
    Variable *root_var = root_scope_->FindVar(name);
    if (root_var == nullptr) {
      continue;
    }
    auto root_tensor = root_var->GetMutable<phi::DenseTensor>();
    Variable *var = thread_scope_->FindVar(name);
    auto tensor = var->Get<phi::DenseTensor>();
    TensorCopy(tensor, root_tensor->place(), root_tensor);
  }
  dev_ctx_->Wait();
#endif
}
template <typename T>
void HogwildWorker::SetZero(phi::DenseTensor *tensor,
                            const phi::DenseTensor &root_tensor) {
  tensor->mutable_data<T>(root_tensor.dims(), place_);
  phi::funcs::set_constant(*dev_ctx_, tensor, static_cast<T>(0));
}

void HogwildWorker::BindingDataFeedMemory() {
  const std::vector<std::string> &input_feed =
      device_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    device_reader_->AddFeedVar(thread_scope_->FindVar(name), name);
  }
}

void HogwildWorker::CreateDeviceResource(const ProgramDesc &main_prog) {
  BuildShardingDepends(main_prog);
  CreateThreadScope(main_prog);
  CreateThreadOperators(main_prog);

#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_GPU_GRAPH)
  float *stat_ptr = sync_stat_.mutable_data<float>(place_, sizeof(float) * 3);
  float flags[] = {0.0, 1.0, 1.0};
  auto stream = static_cast<phi::GPUContext *>(dev_ctx_)->stream();
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(stat_ptr,  // output
                                             &flags,
                                             sizeof(float) * 3,
                                             cudaMemcpyHostToDevice,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#endif
}
// check batch num
bool HogwildWorker::CheckBatchNum(int flag) {
  float ret = 0.0;
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_GPU_GRAPH)
  if (flag > 1) {
    flag = 1;
  } else if (flag < 0) {
    flag = 0;
  }
  //  g_barrier.wait();
  float *stat_ptr = sync_stat_.data<float>();
  auto comm =
      platform::NCCLCommContext::Instance().Get(ring_id_, place_.GetDeviceId());
  //  auto stream = static_cast<phi::GPUContext *>(dev_ctx_)->stream();
  //  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  auto stream = comm->stream();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(&stat_ptr[flag],
                                                              &stat_ptr[2],
                                                              1,
                                                              ncclFloat32,
                                                              ncclProd,
                                                              comm->comm(),
                                                              stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&ret,  // output
                                             &stat_ptr[2],
                                             sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  g_barrier.wait();
#endif
  return (ret > 0.0);
}

bool HogwildWorker::GetPassEnd(int flag) {
  float ret = 0.0;
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_GPU_GRAPH)
  if (flag > 1) {
    flag = 1;
  } else if (flag < 0) {
    flag = 0;
  }
  //  g_barrier.wait();
  float *stat_ptr = sync_stat_.data<float>();
  auto comm =
      platform::NCCLCommContext::Instance().Get(ring_id_, place_.GetDeviceId());
  //  auto stream = static_cast<phi::GPUContext *>(dev_ctx_)->stream();
  //  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  auto stream = comm->stream();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(&stat_ptr[flag],
                                                              &stat_ptr[2],
                                                              1,
                                                              ncclFloat32,
                                                              ncclProd,
                                                              comm->comm(),
                                                              stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&ret,  // output
                                             &stat_ptr[2],
                                             sizeof(float),
                                             cudaMemcpyDeviceToHost,
                                             stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
// g_barrier.wait();
#endif
  return (ret > 0.0);
}

void HogwildWorker::TrainFilesWithProfiler() {
  platform::SetNumThreads(1);
#if defined(PADDLE_WITH_HETERPS) && \
    (defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL))
  platform::SetDeviceId(thread_id_);
#elif defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_XPU_BKCL)
  platform::SetXPUDeviceId(thread_id_);
#endif
  device_reader_->Start();
  std::vector<double> op_total_time;
  op_total_time.resize(ops_.size());
  for (double &op_time : op_total_time) {
    op_time = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  int cur_batch;
  int batch_cnt = 0;
  if (thread_id_ == 0) {
    quit_flag_.store(false);
  }
  g_barrier.wait();
#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_GPU_GRAPH)
  bool train_mode = device_reader_->IsTrainMode();
  bool is_multi_node = false;
  auto gloo = paddle::framework::GlooWrapper::GetInstance();
  if (gloo->Size() > 1) {
    is_multi_node = true;
  }
#endif

  timeline.Start();
  uint64_t total_inst = 0;
#if defined(PADDLE_WITH_GPU_GRAPH) && defined(PADDLE_WITH_HETERPS)
  device_reader_->InitGraphTrainResource();
#endif

  std::unique_ptr<GarbageCollector> gc = nullptr;
  int64_t max_memory_size = GetEagerDeletionThreshold();
  if (max_memory_size >= 0) {
    gc = CreateGarbageCollector(place_, max_memory_size);
  }

  while (1) {
    cur_batch = device_reader_->Next();
#if defined(PADDLE_WITH_GPU_GRAPH)
    if (FLAGS_gpugraph_force_device_batch_num_equal) {
      if (!CheckBatchNum(cur_batch)) {
        break;
      }
    } else if (train_mode && is_multi_node) {
      int pass_end = device_reader_->get_pass_end();
      bool res = GetPassEnd(pass_end);
      VLOG(2) << "reader pass end: " << pass_end
              << ", hogwild worker pass end: " << res;
      if (res) {
        device_reader_->reset_pass_end();
        VLOG(1) << "get all pass end, train pass will exit";
        break;
      }
    } else {
      if (FLAGS_enable_exit_when_partial_worker && train_mode) {
        if (cur_batch <= 0) {
          quit_flag_.store(true, std::memory_order_relaxed);
        }
        g_barrier.wait();
        if (quit_flag_.load(std::memory_order_relaxed) == true) {
          break;
        }
      }
    }
#endif
    if (cur_batch <= 0) {
      break;
    }
    VLOG(3) << "read a batch in thread " << thread_id_;
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();
    for (size_t i = 0; i < ops_.size(); ++i) {
      timeline.Start();
      VLOG(3) << "Going to run op " << op_names_[i];
      ops_[i]->Run(*thread_scope_, place_);
#ifdef PADDLE_WITH_HETERPS
      dev_ctx_->Wait();
#endif
      VLOG(3) << "Op " << op_names_[i] << " Finished";
      timeline.Pause();
      op_total_time[i] += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      if (gc) {
        DeleteUnusedTensors(
            *thread_scope_, ops_[i].get(), unused_vars_, gc.get());
      }
    }

    if (need_dump_field_) {
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && (sharding_mode_ || thread_id_ == 0)) {
      DumpParam(*thread_scope_, batch_cnt);
    }

    total_inst += cur_batch;
    ++batch_cnt;
    PrintFetchVars();
#ifdef PADDLE_WITH_HETERPS
    dev_ctx_->Wait();
    for (size_t i = 0; i < op_names_.size(); ++i) {
      VLOG(1) << "card:" << thread_id_ << ", op: " << op_names_[i]
              << ", mean time: " << op_total_time[i] / total_inst
              << "s, totol time:" << op_total_time[i] << "sec";
    }
#else
    if (thread_id_ == 0) {
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        for (size_t i = 0; i < ops_.size(); ++i) {
          fprintf(stderr,
                  "op_name:[%zu][%s], op_mean_time:[%fs]\n",
                  i,
                  op_names_[i].c_str(),
                  op_total_time[i] / batch_cnt);
        }
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
        fprintf(
            stderr, "%6.2f instances/s\n", total_inst / total_time);  // NOLINT
      }
    }
#endif
    if (gc) {
      gc->DirectClearCallback([this]() { thread_scope_->DropKids(); });
    } else {
      thread_scope_->DropKids();
    }
    timeline.Start();
  }
  VLOG(0) << "GpuPs worker " << thread_id_ << " train cost " << total_time
          << " seconds, ins_num: " << total_inst << " read time: " << read_time
          << "seconds ";

  if (need_dump_field_ || need_dump_param_) {
    writer_.Flush();
  }

#if defined PADDLE_WITH_PSCORE
  if (thread_barrier_) {
    paddle::distributed::Communicator::GetInstance()->BarrierTriggerDecrement();
  }
#endif
}
void HogwildWorker::TrainFiles() {
  platform::SetNumThreads(1);
  platform::Timer timeline;
  timeline.Start();
#if defined(PADDLE_WITH_HETERPS) && \
    (defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL))
  platform::SetDeviceId(thread_id_);
#elif defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_XPU_BKCL)
  platform::SetXPUDeviceId(thread_id_);
#endif

  int total_batch_num = 0;
  // how to accumulate fetched values here
  device_reader_->Start();
  int cur_batch;
  int batch_cnt = 0;
  if (thread_id_ == 0) {
    quit_flag_.store(false);
    // quit_flag_2 = false;
  }
  g_barrier.wait();

#if defined(PADDLE_WITH_HETERPS) && defined(PADDLE_WITH_CUDA)
  platform::SetDeviceId(thread_id_);
#endif
  // while ((cur_batch = device_reader_->Next()) > 0) {
#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_GPU_GRAPH)
  bool is_multi_node = false;
  bool train_mode = device_reader_->IsTrainMode();
  auto gloo = paddle::framework::GlooWrapper::GetInstance();
  if (gloo->Size() > 1) {
    is_multi_node = true;
  }
#endif
#if defined(PADDLE_WITH_GPU_GRAPH) && defined(PADDLE_WITH_HETERPS)
  device_reader_->InitGraphTrainResource();
#endif

  std::unique_ptr<GarbageCollector> gc = nullptr;
  int64_t max_memory_size = GetEagerDeletionThreshold();
  if (max_memory_size >= 0) {
    gc = CreateGarbageCollector(place_, max_memory_size);
  }
  while (1) {
    cur_batch = device_reader_->Next();
#if defined(PADDLE_WITH_GPU_GRAPH)
    if (FLAGS_gpugraph_force_device_batch_num_equal) {
      if (!CheckBatchNum(cur_batch)) {
        break;
      }
    } else if (train_mode && is_multi_node) {
      bool sage_mode = device_reader_->GetSageMode();
      if (!sage_mode) {
        int pass_end = device_reader_->get_pass_end();
        bool res = GetPassEnd(pass_end);
        VLOG(2) << "reader pass end: " << pass_end
                << ", hogwild worker pass end: " << res;
        if (res) {
          device_reader_->reset_pass_end();
          VLOG(1) << "get all pass end, train pass will exit";
         break;
        }
      }
    } else {
      if (FLAGS_enable_exit_when_partial_worker && train_mode) {
        if (cur_batch <= 0) {
          quit_flag_.store(true, std::memory_order_relaxed);
        }
        g_barrier.wait();
        if (quit_flag_.load(std::memory_order_relaxed) == true) {
          break;
        }
      }
    }
#endif
    if (cur_batch <= 0) {
      break;
    }
    for (auto &op : ops_) {
      op->Run(*thread_scope_, place_);
      if (gc) {
        DeleteUnusedTensors(*thread_scope_, op.get(), unused_vars_, gc.get());
      }
    }

    if (need_dump_field_) {
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && (sharding_mode_ || thread_id_ == 0)) {
      DumpParam(*thread_scope_, batch_cnt);
    }

    total_batch_num += cur_batch;
    ++batch_cnt;
    PrintFetchVars();
    if (gc) {
      gc->DirectClearCallback([this]() { thread_scope_->DropKids(); });
    } else {
      thread_scope_->DropKids();
    }
  }
#ifdef PADDLE_WITH_HETERPS
  dev_ctx_->Wait();
#endif
  timeline.Pause();
  VLOG(1) << "worker " << thread_id_ << " train cost " << timeline.ElapsedSec()
          << " seconds, batch_num: " << total_batch_num;

  if (need_dump_field_ || need_dump_param_) {
    writer_.Flush();
  }

#if defined PADDLE_WITH_PSCORE
  if (thread_barrier_) {
    paddle::distributed::Communicator::GetInstance()->BarrierTriggerDecrement();
  }
#endif
}

void HogwildWorker::PrintFetchVars() {
  // call count
  batch_num_++;
  int batch_per_print = fetch_config_.print_period();
  int fetch_var_num = fetch_config_.fetch_var_names_size();

  if (fetch_var_num == 0) {
    return;
  }

  if (thread_id_ == 0 && batch_num_ % batch_per_print == 0) {
    time_t curtime;
    time(&curtime);
    std::array<char, 80> mbstr;
    std::strftime(mbstr.data(),
                  sizeof(mbstr),
                  "%Y-%m-%d %H:%M:%S",
                  std::localtime(&curtime));

    std::stringstream ss;
    ss << "time: [" << mbstr.data() << "], ";
    ss << "batch: [" << batch_num_ << "], ";

    for (int i = 0; i < fetch_var_num; ++i) {
      platform::PrintVar(thread_scope_,
                         fetch_config_.fetch_var_names(i),
                         fetch_config_.fetch_var_str_format(i),
                         &ss);
      if (i < fetch_var_num - 1) {
        ss << ", ";
      }
    }

    std::cout << ss.str() << std::endl;
  }
}

}  // end namespace framework
}  // end namespace paddle
