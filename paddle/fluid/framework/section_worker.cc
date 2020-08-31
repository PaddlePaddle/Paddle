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
#include <float.h>
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/program_desc.h"

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

std::atomic<int> SectionWorker::cpu_id_(0);
std::mutex SectionWorker::thread_mutex;
std::condition_variable SectionWorker::thread_condition;
bool SectionWorker::threads_completed = false;
uint64_t SectionWorker::batch_id_(0);

void SectionWorker::Initialize(const TrainerDesc& desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  program_.reset(new ProgramDesc(
      desc.section_param().section_config(section_id_).program_desc()));
  for (auto& op_desc : program_->Block(0).AllOps()) {
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
  VLOG(3) << "Set " << thread_cpu_id << "th thread affinity to CPU " << proc;
}

void SectionWorker::TrainFiles() {
  VLOG(3) << "begin section_worker TrainFiles";
  AutoSetCPUAffinity(true);

  int64_t max_memory_size = 0;
  std::unique_ptr<GarbageCollector> gc;
  auto unused_vars_ = GetUnusedVars(program_->Block(0), ops_, skip_vars_);
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place_)) {
    if (IsFastEagerDeletionModeEnabled()) {
      gc.reset(new UnsafeFastGPUGarbageCollector(
          BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
    } else {
      gc.reset(new DefaultStreamGarbageCollector(
          BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
    }
  } else if (platform::is_cpu_place(place_)) {
#endif
    gc.reset(new CPUGarbageCollector(
        BOOST_GET_CONST(platform::CPUPlace, place_), max_memory_size));
#ifdef PADDLE_WITH_CUDA
  }
#endif

  if (thread_id_ == 0) {
    while (true) {
      // Start a minibatch.
      for (int i = 0; i < num_microbatches_; ++i) {
        try {
          for (auto& op : ops_) {
            int op_role = op->Attr<int>(std::string("op_role"));
            // We run op with op_role = kLRSched only for the first microbatch
            // to avoid increasing the @LR_DECAY_STEP@ multiple times.
            bool run_first_mbatch =
                op_role == static_cast<int>(OpRole::kForward) ||
                op_role == (static_cast<int>(OpRole::kForward) |
                            static_cast<int>(OpRole::kLoss)) ||
                op_role == static_cast<int>(OpRole::kLRSched);
            bool run_others = op_role == static_cast<int>(OpRole::kForward) ||
                              op_role == (static_cast<int>(OpRole::kForward) |
                                          static_cast<int>(OpRole::kLoss));
            if ((i == 0 && run_first_mbatch) || (i != 0 && run_others)) {
              VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                      << " for scope " << i;
              op->Run(*microbatch_scopes_[i], place_);
              if (gc) {
                DeleteUnusedTensors(*microbatch_scopes_[i], op.get(),
                                    unused_vars_, gc.get());
              }
            }
          }
        } catch (platform::EOFException&) {
          std::unique_lock<std::mutex> lk(thread_mutex);
          threads_completed = true;
          VLOG(3) << "thread " << thread_id_ << " completed.";
          VLOG(3) << "called notify all";
          thread_condition.notify_all();
          VLOG(0) << "EOF encountered";
          return;
        }
        if (i == 0) {
          VLOG(3) << "called notify all";
          std::unique_lock<std::mutex> lk(thread_mutex);
          batch_id_ += 1;
          thread_condition.notify_all();
        }
      }
      // backward pass
      for (int i = 0; i < num_microbatches_; ++i) {
        for (auto& op : ops_) {
          int op_role = op->Attr<int>(std::string("op_role"));
          if (op_role == static_cast<int>(OpRole::kBackward) ||
              op_role == (static_cast<int>(OpRole::kBackward) |
                          static_cast<int>(OpRole::kLoss))) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            op->Run(*microbatch_scopes_[i], place_);
            if (gc) {
              DeleteUnusedTensors(*microbatch_scopes_[i], op.get(),
                                  unused_vars_, gc.get());
            }
          }
        }
      }
      // update pass
      for (auto& op : ops_) {
        int op_role = op->Attr<int>(std::string("op_role"));
        if (op_role == static_cast<int>(OpRole::kOptimize)) {
          VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                  << " for minibatch scope";
          op->Run(*microbatch_scopes_[0], place_);
          if (gc) {
            DeleteUnusedTensors(*microbatch_scopes_[num_microbatches_ - 1],
                                op.get(), unused_vars_, gc.get());
          }
        }
      }
      dev_ctx_->Wait();
    }
  } else {
    while (true) {
      {
        PADDLE_ENFORCE_LE(
            local_batch_id_, batch_id_,
            platform::errors::InvalidArgument(
                "local_batch_id_ (%d) must be less than or equal to "
                "batch_id_ (%d)",
                local_batch_id_, batch_id_));
        std::unique_lock<std::mutex> lk(thread_mutex);
        if (local_batch_id_ == batch_id_ && !threads_completed) {
          thread_condition.wait(lk);
        }
        VLOG(3) << "thread " << thread_id_ << " local_batch_id_ "
                << local_batch_id_ << " batch_id_ " << batch_id_;
        if (threads_completed) {
          VLOG(3) << "thread " << thread_id_ << " completed.";
          lk.unlock();
          threads_completed = false;
          return;
        }
        lk.unlock();
        local_batch_id_ += 1;
      }
      // forward pass:
      for (int i = 0; i < num_microbatches_; ++i) {
        for (auto& op : ops_) {
          int op_role = op->Attr<int>(std::string("op_role"));
          // We run op with op_role = kLRSched only for the first microbatch
          // to avoid increasing the @LR_DECAY_STEP@ multiple times.
          bool run_first_mbatch =
              op_role == static_cast<int>(OpRole::kForward) ||
              op_role == (static_cast<int>(OpRole::kForward) |
                          static_cast<int>(OpRole::kLoss)) ||
              op_role == static_cast<int>(OpRole::kLRSched);
          bool run_others = op_role == static_cast<int>(OpRole::kForward) ||
                            op_role == (static_cast<int>(OpRole::kForward) |
                                        static_cast<int>(OpRole::kLoss));
          if ((i == 0 && run_first_mbatch) || (i != 0 && run_others)) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            op->Run(*microbatch_scopes_[i], place_);
            if (gc) {
              DeleteUnusedTensors(*microbatch_scopes_[i], op.get(),
                                  unused_vars_, gc.get());
            }
          }
        }
      }
      // backward pass
      for (int i = 0; i < num_microbatches_; ++i) {
        for (auto& op : ops_) {
          int op_role = op->Attr<int>(std::string("op_role"));
          if (op_role == static_cast<int>(OpRole::kBackward) ||
              op_role == (static_cast<int>(OpRole::kBackward) |
                          static_cast<int>(OpRole::kLoss))) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            op->Run(*microbatch_scopes_[i], place_);
            if (gc) {
              DeleteUnusedTensors(*microbatch_scopes_[i], op.get(),
                                  unused_vars_, gc.get());
            }
          }
        }
      }
      // update pass
      for (auto& op : ops_) {
        int op_role = op->Attr<int>(std::string("op_role"));
        if (op_role == static_cast<int>(OpRole::kOptimize)) {
          VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                  << " for minibatch scope";
          op->Run(*microbatch_scopes_[0], place_);
          if (gc) {
            DeleteUnusedTensors(*microbatch_scopes_[num_microbatches_ - 1],
                                op.get(), unused_vars_, gc.get());
          }
        }
      }
      dev_ctx_->Wait();
    }
  }
}

void SectionWorker::TrainFilesWithProfiler() {
  VLOG(3) << "begin section_worker TrainFiles with profiler";
  AutoSetCPUAffinity(true);

  platform::Timer batch_timer;
  platform::Timer timeline;

  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  std::vector<double> op_max_time;
  std::vector<double> op_min_time;
  std::vector<uint64_t> op_count;
  for (auto& op : ops_) {
    op_name.push_back(op->Type());
  }
  op_total_time.resize(ops_.size());
  op_max_time.resize(ops_.size());
  op_min_time.resize(ops_.size());
  for (size_t i = 0; i < op_min_time.size(); ++i) {
    op_min_time[i] = DBL_MAX;
  }
  op_count.resize(ops_.size());

  int64_t max_memory_size = 0;
  std::unique_ptr<GarbageCollector> gc;
  // const std::vector<std::string> keep_vars;
  auto unused_vars_ = GetUnusedVars(program_->Block(0), ops_, skip_vars_);
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place_)) {
    if (IsFastEagerDeletionModeEnabled()) {
      gc.reset(new UnsafeFastGPUGarbageCollector(
          BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
    } else {
      gc.reset(new DefaultStreamGarbageCollector(
          BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
    }
  } else if (platform::is_cpu_place(place_)) {
#endif
    gc.reset(new CPUGarbageCollector(
        BOOST_GET_CONST(platform::CPUPlace, place_), max_memory_size));
#ifdef PADDLE_WITH_CUDA
  }
#endif

  if (thread_id_ == 0) {
    while (true) {
      // Start a minibatch.
      // int batch_size = 0;
      batch_timer.Start();
      for (int i = 0; i < num_microbatches_; ++i) {
        try {
          int op_idx = 0;
          for (auto& op : ops_) {
            int op_role = op->Attr<int>(std::string("op_role"));
            // We run op with op_role = kLRSched only for the first microbatch
            // to avoid increasing the @LR_DECAY_STEP@ multiple times.
            bool run_first_mbatch =
                op_role == static_cast<int>(OpRole::kForward) ||
                op_role == (static_cast<int>(OpRole::kForward) |
                            static_cast<int>(OpRole::kLoss)) ||
                op_role == static_cast<int>(OpRole::kLRSched);
            bool run_others = op_role == static_cast<int>(OpRole::kForward) ||
                              op_role == (static_cast<int>(OpRole::kForward) |
                                          static_cast<int>(OpRole::kLoss));
            if ((i == 0 && run_first_mbatch) || (i != 0 && run_others)) {
              VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                      << " for scope " << i;
              timeline.Start();
              op->Run(*microbatch_scopes_[i], place_);
              if (gc) {
                DeleteUnusedTensors(*microbatch_scopes_[i], op.get(),
                                    unused_vars_, gc.get());
              }
              timeline.Pause();
              auto time = timeline.ElapsedUS();
              op_total_time[op_idx] += time;
              if (time > op_max_time[op_idx]) {
                op_max_time[op_idx] = time;
              }
              if (time < op_min_time[op_idx]) {
                op_min_time[op_idx] = time;
              }
              op_count[op_idx] += 1;
              op_total_time[op_idx] += time;
            }
            op_idx++;
          }
        } catch (platform::EOFException&) {
          std::unique_lock<std::mutex> lk(thread_mutex);
          threads_completed = true;
          VLOG(3) << "thread " << thread_id_ << " completed.";
          VLOG(3) << "called notify all";
          thread_condition.notify_all();
          VLOG(0) << "EOF encountered";
          VLOG(0) << "============timeline============";
          for (size_t i = 0; i < ops_.size(); ++i) {
            VLOG(0) << "op: " << op_name[i] << ", max_time: " << op_max_time[i]
                    << ", min_time: " << op_min_time[i]
                    << ", mean_time: " << op_total_time[i] / op_count[i];
          }
          VLOG(0) << "================================";
          return;
        }
        if (i == 0) {
          VLOG(3) << "called notify all";
          std::unique_lock<std::mutex> lk(thread_mutex);
          batch_id_ += 1;
          thread_condition.notify_all();
        }
      }
      // backward pass
      for (int i = 0; i < num_microbatches_; ++i) {
        int op_idx = 0;
        for (auto& op : ops_) {
          int op_role = op->Attr<int>(std::string("op_role"));
          if (op_role == static_cast<int>(OpRole::kBackward) ||
              op_role == (static_cast<int>(OpRole::kBackward) |
                          static_cast<int>(OpRole::kLoss))) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            timeline.Start();
            op->Run(*microbatch_scopes_[i], place_);
            if (gc) {
              DeleteUnusedTensors(*microbatch_scopes_[i], op.get(),
                                  unused_vars_, gc.get());
            }
            timeline.Pause();
            auto time = timeline.ElapsedUS();
            op_total_time[op_idx] += time;
            if (time > op_max_time[op_idx]) {
              op_max_time[op_idx] = time;
            }
            if (time < op_min_time[op_idx]) {
              op_min_time[op_idx] = time;
            }
            op_count[op_idx] += 1;
            op_total_time[op_idx] += time;
          }
          op_idx++;
        }
      }
      // update pass
      int op_idx = 0;
      for (auto& op : ops_) {
        int op_role = op->Attr<int>(std::string("op_role"));
        if (op_role == static_cast<int>(OpRole::kOptimize)) {
          VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                  << " for minibatch scope";
          timeline.Start();
          op->Run(*microbatch_scopes_[0], place_);
          if (gc) {
            DeleteUnusedTensors(*microbatch_scopes_[num_microbatches_ - 1],
                                op.get(), unused_vars_, gc.get());
          }
          timeline.Pause();
          auto time = timeline.ElapsedUS();
          op_total_time[op_idx] += time;
          if (time > op_max_time[op_idx]) {
            op_max_time[op_idx] = time;
          }
          if (time < op_min_time[op_idx]) {
            op_min_time[op_idx] = time;
          }
          op_count[op_idx] += 1;
          op_total_time[op_idx] += time;
        }
        op_idx++;
      }
      dev_ctx_->Wait();
      batch_timer.Pause();
      VLOG(0) << "batch time: " << batch_timer.ElapsedUS();
    }
  } else {
    while (true) {
      {
        PADDLE_ENFORCE_LE(
            local_batch_id_, batch_id_,
            platform::errors::InvalidArgument(
                "local_batch_id_ (%d) must be less than or equal to "
                "batch_id_ (%d)",
                local_batch_id_, batch_id_));
        std::unique_lock<std::mutex> lk(thread_mutex);
        if (local_batch_id_ == batch_id_ && !threads_completed) {
          thread_condition.wait(lk);
        }
        VLOG(3) << "thread " << thread_id_ << " local_batch_id_ "
                << local_batch_id_ << " batch_id_ " << batch_id_;
        if (threads_completed) {
          VLOG(3) << "thread " << thread_id_ << " completed.";
          lk.unlock();
          VLOG(0) << "============timeline============";
          for (size_t i = 0; i < ops_.size(); ++i) {
            VLOG(0) << "op: " << op_name[i] << ", max_time: " << op_max_time[i]
                    << ", min_time: " << op_min_time[i]
                    << ", mean_time: " << op_total_time[i] / op_count[i];
          }
          VLOG(0) << "================================";
          threads_completed = false;
          return;
        }
        lk.unlock();
        local_batch_id_ += 1;
      }
      // forward pass:
      for (int i = 0; i < num_microbatches_; ++i) {
        int op_idx = 0;
        for (auto& op : ops_) {
          int op_role = op->Attr<int>(std::string("op_role"));
          // We run op with op_role = kLRSched only for the first microbatch
          // to avoid increasing the @LR_DECAY_STEP@ multiple times.
          bool run_first_mbatch =
              op_role == static_cast<int>(OpRole::kForward) ||
              op_role == (static_cast<int>(OpRole::kForward) |
                          static_cast<int>(OpRole::kLoss)) ||
              op_role == static_cast<int>(OpRole::kLRSched);
          bool run_others = op_role == static_cast<int>(OpRole::kForward) ||
                            op_role == (static_cast<int>(OpRole::kForward) |
                                        static_cast<int>(OpRole::kLoss));
          if ((i == 0 && run_first_mbatch) || (i != 0 && run_others)) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            timeline.Start();
            op->Run(*microbatch_scopes_[i], place_);
            if (gc) {
              DeleteUnusedTensors(*microbatch_scopes_[i], op.get(),
                                  unused_vars_, gc.get());
            }
            timeline.Pause();
            auto time = timeline.ElapsedUS();
            op_total_time[op_idx] += time;
            if (time > op_max_time[op_idx]) {
              op_max_time[op_idx] = time;
            }
            if (time < op_min_time[op_idx]) {
              op_min_time[op_idx] = time;
            }
            op_count[op_idx] += 1;
            op_total_time[op_idx] += time;
          }
          op_idx++;
        }
      }
      // backward pass
      for (int i = 0; i < num_microbatches_; ++i) {
        int op_idx = 0;
        for (auto& op : ops_) {
          int op_role = op->Attr<int>(std::string("op_role"));
          if (op_role == static_cast<int>(OpRole::kBackward) ||
              op_role == (static_cast<int>(OpRole::kBackward) |
                          static_cast<int>(OpRole::kLoss))) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            timeline.Start();
            op->Run(*microbatch_scopes_[i], place_);
            if (gc) {
              DeleteUnusedTensors(*microbatch_scopes_[i], op.get(),
                                  unused_vars_, gc.get());
            }
            timeline.Pause();
            auto time = timeline.ElapsedUS();
            op_total_time[op_idx] += time;
            if (time > op_max_time[op_idx]) {
              op_max_time[op_idx] = time;
            }
            if (time < op_min_time[op_idx]) {
              op_min_time[op_idx] = time;
            }
            op_count[op_idx] += 1;
            op_total_time[op_idx] += time;
          }
          op_idx++;
        }
      }
      // update pass
      int op_idx = 0;
      for (auto& op : ops_) {
        int op_role = op->Attr<int>(std::string("op_role"));
        if (op_role == static_cast<int>(OpRole::kOptimize)) {
          VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                  << " for minibatch scope";
          timeline.Start();
          op->Run(*microbatch_scopes_[0], place_);
          if (gc) {
            DeleteUnusedTensors(*microbatch_scopes_[num_microbatches_ - 1],
                                op.get(), unused_vars_, gc.get());
          }
          timeline.Pause();
          auto time = timeline.ElapsedUS();
          op_total_time[op_idx] += time;
          if (time > op_max_time[op_idx]) {
            op_max_time[op_idx] = time;
          }
          if (time < op_min_time[op_idx]) {
            op_min_time[op_idx] = time;
          }
          op_count[op_idx] += 1;
          op_total_time[op_idx] += time;
        }
        op_idx++;
      }
      dev_ctx_->Wait();
    }
  }
}
}  // namespace framework
}  // namespace paddle
#endif
