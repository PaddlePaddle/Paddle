/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_NCCL)
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

namespace paddle {
namespace framework {

uint64_t SyncFunctor::sync_flag_ = 0;
std::vector<Scope*> SyncFunctor::pipeline_scopes_;

SyncFunctor::SyncFunctor(int rank_id, int rank_num, int sync_steps)
    : rank_id_(rank_id), rank_num_(rank_num), sync_steps_(sync_steps) {
  PADDLE_ENFORCE(rank_num > 1, "rank_num should larger than 1");
  counter_ = 0;
  sync_signal_ = 0;
  uint8_t* ptr = reinterpret_cast<uint8_t*>(&sync_signal_);
  for (int i = 0; i < rank_num_; ++i) {
    ptr[i] = 0xFF;
  }
}

int SyncFunctor::operator()(Scope* scope) {
  ++counter_;
  if (counter_ < sync_steps_) {
    return 0;
  }
  if (counter_ == sync_steps_) {
    reinterpret_cast<uint8_t*>(&sync_flag_)[rank_id_] = 0xFF;
  }

  if (sync_flag_ == sync_signal_) {
    static std::mutex mutex;
    if (mutex.try_lock()) {
      if (sync_flag_ == sync_signal_) {
        Synchronize();
        sync_flag_ = 0;
      }
      mutex.unlock();
    }
  }

  if (sync_flag_ == 0) {
    counter_ = 0;
  }
  return 0;
}

void SyncFunctor::Synchronize() {
  for (const std::string& name : *sync_param_) {
    platform::NCCLGroupGuard guard;
    for (int i = 0; i < rank_num_; ++i) {
      const platform::NCCLContext& nccl_ctx = nccl_ctx_map_->at(i);
      LoDTensor* tensor =
          pipeline_scopes_[i]->Var(name)->GetMutable<LoDTensor>();
      // TODO(hutuxian): do not depend on data type explicitly
      float* data =
          tensor->mutable_data<float>(nccl_ctx_map_->DevCtx(i)->GetPlace());
      const int numel = tensor->numel();

      paddle::framework::AttributeMap attrs;
      attrs.insert({"scale", static_cast<float>(1. / rank_num_)});
      auto scale_op = framework::OpRegistry::CreateOp("scale", {{"X", {name}}},
                                                      {{"Out", {name}}}, attrs);
      scale_op->Run(*(pipeline_scopes_[i]),
                    nccl_ctx_map_->DevCtx(i)->GetPlace());
      PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
          data, data, numel, ncclFloat, ncclSum, nccl_ctx.comm(),
          dynamic_cast<platform::CUDADeviceContext*>(
              platform::DeviceContextPool::Instance().Get(
                  platform::CUDAPlace(i)))
              ->stream()));
    }
  }
  nccl_ctx_map_->WaitAll();
}

std::atomic<int> SectionWorker::cpu_id_(0);
void SectionWorker::Initialize(const TrainerDesc& trainer_desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  std::shared_ptr<framework::ProgramDesc> program;
  program.reset(new ProgramDesc(
      trainer_desc.section_param().section_config(section_id_).program_desc()));
  for (auto& op_desc : program->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
}

void SectionWorker::AutoSetCPUAffinity(bool reuse) {
  int thread_cpu_id = cpu_id_.fetch_add(1);

  unsigned concurrency_cap = std::thread::hardware_concurrency();
  unsigned proc = thread_cpu_id;

  if (proc >= concurrency_cap) {
    if (reuse) {
      proc %= concurrency_cap;
    } else {
      LOG(INFO) << "All " << concurrency_cap
                << " CPUs have been set affinities. Fail to set "
                << thread_cpu_id << "th thread";
      return;
    }
  }

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(proc, &mask);

  if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
    LOG(WARNING) << "Fail to set thread affinity to CPU " << proc;
    return;
  }

  CPU_ZERO(&mask);
  if ((0 != sched_getaffinity(0, sizeof(mask), &mask)) ||
      (0 == CPU_ISSET(proc, &mask))) {
    LOG(WARNING) << "Fail to set thread affinity to CPU " << proc;
  }
  SEC_LOG << "Set " << thread_cpu_id << "th thread affinity to CPU " << proc;
}

void SectionWorker::TrainFiles() {
  SEC_LOG << "begin section_worker TrainFiles";
  AutoSetCPUAffinity(true);

  int64_t step_cnt = 0;
  int64_t accum_num = 0;
  int batch_size = 0;
  Scope* scope = nullptr;
  if (device_reader_ != nullptr) {
    device_reader_->Start();
  }
  while (in_scope_queue_->Receive(&scope)) {
    if (device_reader_ != nullptr) {
      device_reader_->AssignFeedVar(*scope);
      batch_size = device_reader_->Next();
      if (batch_size <= 0) {
        break;
      }
      SEC_LOG << "read batch size: " << batch_size;
    } else {
      // TODO(hutuxian): Keep batch_size in scope? Or is there a better way to
      // fetch batch_size? Some variables may not have batch_size.
      PADDLE_ENFORCE(
          in_var_names_->size(),
          "Section without a reader or in variable is not supported by now");
      const LoDTensor& tensor =
          scope->FindVar(in_var_names_->at(0))->Get<LoDTensor>();
      batch_size =
          tensor.lod().size() ? tensor.lod()[0].size() - 1 : tensor.dims()[0];
      SEC_LOG << "input batch size: " << batch_size;
    }

    Scope* exe_scope = scope;
    if (section_id_ > 0 && platform::is_gpu_place(place_)) {
      SEC_LOG << "CPU2GPU memory copy";

      if (scope->kids().empty()) {
        exe_scope = &scope->NewScope();
      } else {
        exe_scope = scope->kids().front();
        PADDLE_ENFORCE(scope->kids().size() == 1, "scope->kids().size(): %zu",
                       scope->kids().size());
      }

      for (const std::string& name : *in_var_names_) {
        const LoDTensor& src_tensor = scope->FindVar(name)->Get<LoDTensor>();
        if (platform::is_gpu_place(src_tensor.place())) {
          continue;
        }
        LoDTensor* gpu_tensor = exe_scope->Var(name)->GetMutable<LoDTensor>();
        gpu_tensor->set_lod(src_tensor.lod());
        TensorCopy(*static_cast<const Tensor*>(&src_tensor), place_, *dev_ctx_,
                   static_cast<Tensor*>(gpu_tensor));
      }
    }

    SEC_LOG << "begin running ops";

    for (auto& op : ops_) {
      op->Run(*exe_scope, place_);
    }
    exe_scope->DropKids();
    // Wait for GPU calc finising, as the cudaMemcpy and GPU calc may be in
    // different streams
    // No effect when it is a CPUDeviceContext
    dev_ctx_->Wait();

#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    auto& metric_list = box_ptr->GetMetricList();
    for (auto iter = metric_list.begin(); iter != metric_list.end(); iter++) {
      auto* metric_msg = iter->second;
      if (metric_msg->IsJoin() != box_ptr->PassFlag()) {
        continue;
      }
      metric_msg->add_data(exe_scope);
    }
#endif
    if (section_id_ != section_num_ - 1 && platform::is_gpu_place(place_)) {
      // FIXME: Temporarily we assume two adjacent sections are in different
      // places,
      // and we do data transformation only in sections in GPU place, so the
      // data is
      // transform from GPU to CPU
      // A better way to handle such a data transformation is to record each
      // place of
      // joint-out variables, and do transform as required

      SEC_LOG << "GPU2CPU memory copy";

      for (const std::string& name : *out_var_names_) {
        const LoDTensor& src_tensor =
            exe_scope->FindVar(name)->Get<LoDTensor>();
        LoDTensor* dst_tensor = scope->Var(name)->GetMutable<LoDTensor>();
        dst_tensor->set_lod(src_tensor.lod());
        TensorCopy(*static_cast<const Tensor*>(&src_tensor),
                   next_section_place_, *dev_ctx_,
                   static_cast<Tensor*>(dst_tensor));
      }
    }

    out_scope_queue_->Send(scope);

    if (sync_func_) {
      (*sync_func_)(scope);
    }

    ++step_cnt;
    accum_num += batch_size;
  }

  worker_count_mutex_->lock();
  --(*worker_count_);
  worker_count_mutex_->unlock();

  if (*worker_count_ <= 0) {
    while (section_id_ < section_num_ - 1 && out_scope_queue_->Size()) {
      sleep(1);
    }
    out_scope_queue_->Close();
  }
}

void SectionWorker::TrainFilesWithProfiler() {
  SEC_LOG << "begin section_worker TrainFiles with profiler";
  AutoSetCPUAffinity(true);

  int64_t step_cnt = 0;
  int64_t accum_num = 0;
  int batch_size = 0;
  Scope* scope = nullptr;

  platform::Timer reader_timer;
  platform::Timer cal_timer;
  platform::Timer trans_timer;
  platform::Timer sync_timer;
  platform::Timer main_timer;
  platform::Timer outer_timer;

  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    op_name.push_back(op->Type());
  }
  op_total_time.resize(ops_.size());
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  if (device_reader_ != nullptr) {
    device_reader_->Start();
  }

  bool started = false;
  while (in_scope_queue_->Receive(&scope)) {
    if (UNLIKELY(!started)) {
      outer_timer.Start();
      started = true;
    }
    main_timer.Resume();

    if (device_reader_ != nullptr) {
      reader_timer.Resume();
      device_reader_->AssignFeedVar(*scope);
      batch_size = device_reader_->Next();
      reader_timer.Pause();
      if (batch_size <= 0) {
        break;
      }
      SEC_LOG << "read batch size: " << batch_size;
    } else {
      PADDLE_ENFORCE(
          in_var_names_->size(),
          "Section without a reader or in variable is not supported by now");
      const LoDTensor& tensor =
          scope->FindVar(in_var_names_->at(0))->Get<LoDTensor>();
      batch_size =
          tensor.lod().size() ? tensor.lod()[0].size() - 1 : tensor.dims()[0];
      SEC_LOG << "input batch size: " << batch_size;
    }

    Scope* exe_scope = scope;
    if (section_id_ > 0 && platform::is_gpu_place(place_)) {
      SEC_LOG << "CPU2GPU memory copy";
      trans_timer.Resume();
      if (scope->kids().empty()) {
        exe_scope = &scope->NewScope();
      } else {
        exe_scope = scope->kids().front();
        PADDLE_ENFORCE(scope->kids().size() == 1, "scope->kids().size(): %zu",
                       scope->kids().size());
      }

      for (const std::string& name : *in_var_names_) {
        const LoDTensor& src_tensor = scope->FindVar(name)->Get<LoDTensor>();
        if (platform::is_gpu_place(src_tensor.place())) {
          continue;
        }
        LoDTensor* gpu_tensor = exe_scope->Var(name)->GetMutable<LoDTensor>();
        gpu_tensor->set_lod(src_tensor.lod());
        TensorCopy(*static_cast<const Tensor*>(&src_tensor), place_, *dev_ctx_,
                   static_cast<Tensor*>(gpu_tensor));
      }
      trans_timer.Pause();
    }

    SEC_LOG << "begin running ops";
    cal_timer.Resume();
    int op_id = 0;
    dev_ctx_->Wait();
    for (auto& op : ops_) {
      timeline.Start();
      op->Run(*exe_scope, place_);
      dev_ctx_->Wait();
      timeline.Pause();
      op_total_time[op_id++] += timeline.ElapsedUS();
    }
    exe_scope->DropKids();
    // Wait for GPU calc finising, as the cudaMemcpy and GPU calc may be in
    // different streams
    // No effect when it is a CPUDeviceContext
    dev_ctx_->Wait();
    cal_timer.Pause();
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    auto& metric_list = box_ptr->GetMetricList();
    for (auto iter = metric_list.begin(); iter != metric_list.end(); iter++) {
      auto* metric_msg = iter->second;
      if (metric_msg->IsJoin() != box_ptr->PassFlag()) {
        continue;
      }
      metric_msg->add_data(exe_scope);
    }
#endif

    if (section_id_ != section_num_ - 1 && platform::is_gpu_place(place_)) {
      // FIXME: Temporarily we assume two adjacent sections are in different
      // places,
      // and we do data transformation only in sections in GPU place, so the
      // data is
      // transform from GPU to CPU
      // A better way to handle such a data transformation is to record each
      // place of
      // joint-out variables, and do transform as required

      SEC_LOG << "GPU2CPU memory copy";
      trans_timer.Resume();
      for (const std::string& name : *out_var_names_) {
        const LoDTensor& src_tensor =
            exe_scope->FindVar(name)->Get<LoDTensor>();
        LoDTensor* dst_tensor = scope->Var(name)->GetMutable<LoDTensor>();
        dst_tensor->set_lod(src_tensor.lod());
        TensorCopy(*static_cast<const Tensor*>(&src_tensor),
                   next_section_place_, *dev_ctx_,
                   static_cast<Tensor*>(dst_tensor));
      }
      trans_timer.Pause();
    }

    out_scope_queue_->Send(scope);

    if (sync_func_) {
      sync_timer.Resume();
      (*sync_func_)(scope);
      sync_timer.Pause();
    }

    ++step_cnt;
    accum_num += batch_size;
    main_timer.Pause();
  }
  outer_timer.Pause();

  worker_count_mutex_->lock();
  --(*worker_count_);
  worker_count_mutex_->unlock();

  if (*worker_count_ <= 0) {
    while (section_id_ < section_num_ - 1 && out_scope_queue_->Size()) {
      sleep(1);
    }
    out_scope_queue_->Close();
  }
  LOG(ERROR) << "log_for_profile"
             << " card:" << pipeline_id_ << " thread:" << thread_id_
             << " section:" << section_id_ << " step_count:" << step_cnt
             << " batch_count:" << accum_num
             << " read_time:" << reader_timer.ElapsedUS()
             << " trans_time:" << trans_timer.ElapsedUS()
             << " cal_time:" << cal_timer.ElapsedUS()
             << " sync_time:" << sync_timer.ElapsedUS()
             << " main_time:" << main_timer.ElapsedUS()
             << " outer_time:" << outer_timer.ElapsedUS();
  for (size_t i = 0; i < ops_.size(); ++i) {
    LOG(ERROR) << "op: " << op_name[i]
               << ", mean time: " << op_total_time[i] / accum_num;
  }
}
}  // namespace framework
}  // namespace paddle
#endif
