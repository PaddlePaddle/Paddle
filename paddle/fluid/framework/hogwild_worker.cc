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
#include "paddle/fluid/operators/controlflow/conditional_block_op_helper.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/lodtensor_printer.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/flags.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
PHI_DECLARE_bool(dynamic_static_unified_comm);
#endif

#if defined PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

PHI_DECLARE_bool(enable_exit_when_partial_worker);

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
    stat_var_name_map_[param_.stat_var_names(i)] = 1;
  }
}

void HogwildWorker::CreateThreadOperators(const ProgramDesc &program) {
  auto &block = program.Block(0);
  op_names_.clear();
  for (auto &op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    op_names_.push_back(op_desc->Type());
    OperatorBase *local_op_ptr = local_op.release();
    ops_.push_back(local_op_ptr);
    continue;
  }
  operators::PrepareSafeEagerDeletionOnConditionalOpAndConditionalGradOp(
      program, 0, ops_);
}

void HogwildWorker::CreateThreadScope(const ProgramDesc &program) {
  auto &block = program.Block(0);

  PADDLE_ENFORCE_NOT_NULL(
      root_scope_,
      platform::errors::NotFound(
          "Root scope should be set before creating thread scope."));

  thread_scope_ = &root_scope_->NewScope();

  for (auto &var : block.AllVars()) {
    all_param_.push_back(var->Name());
    if (var->Persistable()) {
      auto *ptr = root_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      if (stat_var_name_map_.find(var->Name()) != stat_var_name_map_.end() &&
          thread_id_ != 0) {
        int tensor_dim = static_cast<int>(root_scope_->FindVar(var->Name())
                                              ->GetMutable<phi::DenseTensor>()
                                              ->numel());
        auto *ptr1 = thread_scope_->Var(var->Name());
        InitializeVariable(ptr1, var->GetType());
        phi::DenseTensor *thread_tensor = ptr1->GetMutable<phi::DenseTensor>();
        phi::DenseTensor *root_tensor =
            root_scope_->FindVar(var->Name())->GetMutable<phi::DenseTensor>();
#define MemsetCallback(cpp_type, proto_type)                                  \
  do {                                                                        \
    if (framework::TransToProtoVarType(root_tensor->dtype()) == proto_type) { \
      SetZero<cpp_type>(thread_tensor, root_tensor, tensor_dim);              \
    }                                                                         \
  } while (0)
        _ForEachDataType_(MemsetCallback);
      }
    } else {
      auto *ptr = thread_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
    }
  }
}

template <typename T>
void HogwildWorker::SetZero(phi::DenseTensor *tensor,
                            phi::DenseTensor *root_tensor,
                            int tensor_dim) {
  T *ptr = tensor->mutable_data<T>(root_tensor->dims(), platform::CPUPlace());
  memset(ptr, 0, sizeof(T) * tensor_dim);
}

void HogwildWorker::BindingDataFeedMemory() {
  const std::vector<std::string> &input_feed =
      device_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    device_reader_->AddFeedVar(thread_scope_->FindVar(name), name);
  }
}

void HogwildWorker::CreateDeviceResource(const ProgramDesc &main_prog) {
  CreateThreadScope(main_prog);
  CreateThreadOperators(main_prog);

#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_GPU_GRAPH)
  float *stat_ptr = sync_stat_.mutable_data<float>(place_, sizeof(float) * 3);
  float flags[] = {0.0, 1.0, 0.0};
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
  g_barrier.wait();
  float *stat_ptr = sync_stat_.data<float>();
  int nranks = 0;
  int ring_id = 0;
  platform::NCCLComm *comm = nullptr;
  const auto &comm_context_manager =
      phi::distributed::CommContextManager::GetInstance();
  phi::distributed::NCCLCommContext *comm_ctx = nullptr;
  if (FLAGS_dynamic_static_unified_comm) {
    PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id)),
                      true,
                      platform::errors::InvalidArgument(
                          "You choose to use new communication library by "
                          "setting environment "
                          "variable FLAGS_dynamic_static_unified_comm True. "
                          "But ring_id(%d) is "
                          "not found in comm_context_manager.",
                          std::to_string(ring_id)));
    comm_ctx = static_cast<phi::distributed::NCCLCommContext *>(
        comm_context_manager.Get(std::to_string(ring_id)));
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      platform::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
    nranks = comm_ctx->GetSize();
  } else {
    comm = platform::NCCLCommContext::Instance().Get(ring_id,
                                                     place_.GetDeviceId());
    nranks = comm->nranks();
  }

  auto stream = static_cast<phi::GPUContext *>(dev_ctx_)->stream();
  if (comm_ctx) {
    // comm_ctx->AllReduce only support allreduce on the whole tensor,
    // single element is not supported now.
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::ncclAllReduce(&stat_ptr[flag],
                                         &stat_ptr[2],
                                         1,
                                         ncclFloat32,
                                         ncclProd,
                                         comm_ctx->GetNcclComm(),
                                         stream));

  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(&stat_ptr[flag],
                                                                &stat_ptr[2],
                                                                1,
                                                                ncclFloat32,
                                                                ncclProd,
                                                                comm->comm(),
                                                                stream));
  }

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
  std::vector<std::string> op_name;
  for (auto &op : ops_) {
    op_name.push_back(op->Type());
  }
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
  while (true) {
    cur_batch = device_reader_->Next();
#if defined(PADDLE_WITH_GPU_GRAPH)
    if (is_multi_node) {
      if (!CheckBatchNum(cur_batch)) {
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
      bool need_skip = false;
      for (auto &skip_op : skip_ops_) {
        if (ops_[i]->Type().find(skip_op) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      timeline.Start();
      VLOG(3) << "Going to run op " << op_name[i];
      if (!need_skip) {
        ops_[i]->Run(*thread_scope_, place_);
#ifdef PADDLE_WITH_HETERPS
        dev_ctx_->Wait();
#endif
      }
      VLOG(3) << "Op " << op_name[i] << " Finished";
      timeline.Pause();
      op_total_time[i] += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
    }

    if (need_dump_field_) {
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && thread_id_ == 0) {
      DumpParam(*thread_scope_, batch_cnt);
    }

    total_inst += cur_batch;
    ++batch_cnt;
    PrintFetchVars();
#ifdef PADDLE_WITH_HETERPS
    dev_ctx_->Wait();
    for (size_t i = 0; i < op_name.size(); ++i) {
      VLOG(1) << "card:" << thread_id_ << ", op: " << op_name[i]
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
                  op_name[i].c_str(),
                  op_total_time[i] / batch_cnt);
        }
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
        fprintf(
            stderr, "%6.2f instances/s\n", total_inst / total_time);  // NOLINT
      }
    }
#endif
    thread_scope_->DropKids();
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
  while (true) {
    cur_batch = device_reader_->Next();
#if defined(PADDLE_WITH_GPU_GRAPH)
    if (is_multi_node) {
      if (!CheckBatchNum(cur_batch)) {
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
    for (auto &op : ops_) {
      bool need_skip = false;
      for (auto &skip_op : skip_ops_) {
        if (op->Type().find(skip_op) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        op->Run(*thread_scope_, place_);
      }
    }

    if (need_dump_field_) {
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && thread_id_ == 0) {
      DumpParam(*thread_scope_, batch_cnt);
    }

    total_batch_num += cur_batch;
    ++batch_cnt;
    PrintFetchVars();
    thread_scope_->DropKids();
#ifdef PADDLE_WITH_HETERPS
    dev_ctx_->Wait();
#endif
  }
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
