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
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

namespace paddle {
namespace framework {

std::atomic<int> ModelParallelWorker::cpu_id_(0);
std::mutex ModelParallelWorker::thread_mutex;
std::condition_variable ModelParallelWorker::thread_condition;
bool ModelParallelWorker::threads_completed = false;
uint64_t ModelParallelWorker::batch_id_(0);

void ModelParallelWorker::Initialize(const TrainerDesc& trainer_desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  program_.reset(new ProgramDesc(
      trainer_desc.section_param().section_config(section_id_).program_desc()));
  for (auto& op_desc : program_->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
}

void ModelParallelWorker::AutoSetCPUAffinity(bool reuse) {
  int thread_cpu_id = cpu_id_.fetch_add(1);

  unsigned concurrency_cap = std::thread::hardware_concurrency();
  unsigned proc = thread_cpu_id;

  if (proc >= concurrency_cap) {
    if (reuse) {
      proc %= concurrency_cap;
    } else {
      LOG(INFO) << "All " << concurrency_cap
                << " CPUs have been set affinities. Fail to set the "
                << thread_id_ << "th thread.";
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
  VLOG(3) << "Set " << thread_id_ << "th thread affinity to CPU " << proc;
}

void ModelParallelWorker::ForwardPass(int macrobatch_id) {
  for (auto& op : ops_) {
    int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
    // We run op with op_role = kLRSched only for the first macrobatch
    // to avoid increasing the @LR_DECAY_STEP@ multiple times.
    bool run_first_mbatch = op_role == static_cast<int>(OpRole::kForward) ||
                            op_role == (static_cast<int>(OpRole::kForward) |
                                        static_cast<int>(OpRole::kLoss)) ||
                            op_role == static_cast<int>(OpRole::kLRSched);
    bool run_others = op_role == static_cast<int>(OpRole::kForward) ||
                      op_role == (static_cast<int>(OpRole::kForward) |
                                  static_cast<int>(OpRole::kLoss));
    if ((macrobatch_id == 0 && run_first_mbatch) ||
        (macrobatch_id != 0 && run_others)) {
      VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
              << " for scope " << macrobatch_id;
      op->Run(*macrobatch_scopes_[macrobatch_id], place_);
    }
  }
}

void ModelParallelWorker::BackwardPass(int macrobatch_id) {
  for (auto& op : ops_) {
    int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
    if (op_role == static_cast<int>(OpRole::kBackward) ||
        op_role == (static_cast<int>(OpRole::kBackward) |
                    static_cast<int>(OpRole::kLoss))) {
      VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
              << " for scope " << macrobatch_id;
      op->Run(*macrobatch_scopes_[macrobatch_id], place_);
    }
  }
}

void ModelParallelWorker::OptimizePass() {
  for (auto& op : ops_) {
    int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
    if (op_role == static_cast<int>(OpRole::kOptimize)) {
      VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
              << " for minibatch scope";
      // op->Run(*minibatch_scope_, place_);
      op->Run(*macrobatch_scopes_[num_macrobatches_ - 1], place_);
    }
  }
}

void ModelParallelWorker::ForwardPassProfile(int macrobatch_id) {
  for (auto& op : ops_) {
    int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
    // We run op with op_role = kLRSched only for the first macrobatch
    // to avoid increasing the @LR_DECAY_STEP@ multiple times.
    bool run_first_mbatch = op_role == static_cast<int>(OpRole::kForward) ||
                            op_role == (static_cast<int>(OpRole::kForward) |
                                        static_cast<int>(OpRole::kLoss)) ||
                            op_role == static_cast<int>(OpRole::kLRSched);
    bool run_others = op_role == static_cast<int>(OpRole::kForward) ||
                      op_role == (static_cast<int>(OpRole::kForward) |
                                  static_cast<int>(OpRole::kLoss));
    if ((macrobatch_id == 0 && run_first_mbatch) ||
        (macrobatch_id != 0 && run_others)) {
      VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
              << " for scope " << macrobatch_id;
      op->Run(*macrobatch_scopes_[macrobatch_id], place_);
    }
  }
}

void ModelParallelWorker::BackwardPassProfile(int macrobatch_id) {
  for (auto& op : ops_) {
    int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
    if (op_role == static_cast<int>(OpRole::kBackward) ||
        op_role == (static_cast<int>(OpRole::kBackward) |
                    static_cast<int>(OpRole::kLoss))) {
      VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
              << " for scope " << macrobatch_id;
      op->Run(*macrobatch_scopes_[macrobatch_id], place_);
    }
  }
}

void ModelParallelWorker::OptimizePassProfile() {
  for (auto& op : ops_) {
    int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
    if (op_role == static_cast<int>(OpRole::kOptimize)) {
      VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
              << " for minibatch scope";
      // op->Run(*minibatch_scope_, place_);
      op->Run(*macrobatch_scopes_[num_macrobatches_ - 1], place_);
    }
  }
}

void ModelParallelWorker::TrainFiles() {
  AutoSetCPUAffinity(true);

  if (thread_id_ == 0) {
    // For the first section.
    PADDLE_ENFORCE_NOT_NULL(device_reader_,
                            "The device reader for the first "
                            "thread should not be null.");
    device_reader_->Start();
    device_reader_->AssignFeedVar(*minibatch_scope_);
    while (true) {
      // Start a minibatch.
      int batch_size = 0;
      for (int i = 0; i < num_macrobatches_; ++i) {
        batch_size = device_reader_->Next();
        {
          std::unique_lock<std::mutex> lk(thread_mutex);
          if (batch_size <= 0) {
            threads_completed = true;
            VLOG(3) << "thread " << thread_id_ << " completed.";
            thread_condition.notify_all();
            return;
          }
          if (i == 0) {
            batch_id_ += 1;
            thread_condition.notify_all();
          }
        }
        // forward pass:
        ForwardPass(i);
      }
      // backward pass
      for (int i = 0; i < num_macrobatches_; ++i) {
        BackwardPass(i);
      }
      // update pass
      OptimizePass();
      dev_ctx_->Wait();
    }
  } else {
    while (true) {
      {
        std::unique_lock<std::mutex> lk(thread_mutex);
        PADDLE_ENFORCE_LE(local_batch_id_, batch_id_,
                          "local_batch_id_ (%d) must be less than or equal to "
                          "batch_id_ (%d)",
                          local_batch_id_, batch_id_);
        if (local_batch_id_ == batch_id_ && !threads_completed) {
          thread_condition.wait(lk);
        }
        VLOG(3) << "thread " << thread_id_ << " local_batch_id_ "
                << local_batch_id_ << " batch_id_ " << batch_id_;
        if (threads_completed) {
          VLOG(3) << "thread " << thread_id_ << " completed.";
          lk.unlock();
          return;
        }
        lk.unlock();
        local_batch_id_ += 1;
      }
      // forward pass:
      for (int i = 0; i < num_macrobatches_; ++i) {
        ForwardPass(i);
      }
      // backward pass
      for (int i = 0; i < num_macrobatches_; ++i) {
        BackwardPass(i);
      }
      // update pass
      OptimizePass();
      dev_ctx_->Wait();
    }
  }
}

void ModelParallelWorker::TrainFilesWithProfiler() {
  VLOG(3) << "ModelParallelWorker::TrainFilesWithProfiler";
  AutoSetCPUAffinity(true);

  platform::Timer reader_timer;
  platform::Timer calc_timer;

  reader_timer.Resume();
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
    op_min_time[i] = 100000000;
  }
  op_count.resize(ops_.size());

#if 0
  int64_t max_memory_size = 0;
  std::unique_ptr<GarbageCollector> gc;
  // const std::vector<std::string> keep_vars;
  auto unused_vars_ = GetUnusedVars(program_->Block(0), ops_, skip_vars_);
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(place_)) {
    if (IsFastEagerDeletionModeEnabled()) {
      gc.reset(new UnsafeFastGPUGarbageCollector(
          boost::get<platform::CUDAPlace>(place_), max_memory_size));
    } else {
      gc.reset(new DefaultStreamGarbageCollector(
          boost::get<platform::CUDAPlace>(place_), max_memory_size));
    }
  } else if (platform::is_cpu_place(place_)) {
#endif
    gc.reset(new CPUGarbageCollector(boost::get<platform::CPUPlace>(place_),
                                     max_memory_size));
#ifdef PADDLE_WITH_CUDA
  }
#endif
#endif

  calc_timer.Resume();

  if (thread_id_ == 0) {
    // For the first section.
    // PADDLE_ENFORCE_NOT_NULL(device_reader_,
    //                         "The device reader for the first "
    //                         "thread should not be null.");
    // device_reader_->Start();
    // device_reader_->AssignFeedVar(*minibatch_scope_);

    // Get the lod blocking queue
    // paddle::operators::reader::LoDTensorBlockingQueueHolder *queue_holder;
    // for (auto& op : ops_) {
    //  if (op->Type() == "create_py_reader") {
    //    auto queue_name = op->Input("blocking_queue");
    //    VLOG(3) << "blocking_queue:" << queue_name;
    //    queue_holder =
    //    macrobatch_scopes_[0]->FindVar(queue_name)->GetMutable<paddle::operators::reader::LoDTensorBlockingQueueHolder>();
    //    break;
    //  }
    //}
    // auto queue = queue_holder->GetQueue();
    while (true) {
      // Start a minibatch.
      // int batch_size = 0;
      timeline.Start();
      for (int i = 0; i < num_macrobatches_; ++i) {
        // reader_timer.Resume();
        // batch_size = device_reader_->Next();
        // reader_timer.Pause();
        // while (!queue->IsClosed() && queue->Size() == 0) ;
        // if (queue->IsClosed()) {
        //  std::unique_lock<std::mutex> lk(thread_mutex);
        //  threads_completed = true;
        //  VLOG(3) << "thread " << thread_id_ << " completed.";
        //  VLOG(3) << "called notify all";
        //  thread_condition.notify_all();
        //  calc_timer.Pause();
        //  VLOG(0) << "read_time: " << reader_timer.ElapsedUS();
        //  VLOG(0) << "calc_time: " << calc_timer.ElapsedUS();
        //  return;
        //}

        //{
        //  std::unique_lock<std::mutex> lk(thread_mutex);
        //  //if (batch_size <= 0) {
        //  //  threads_completed = true;
        //  //  VLOG(3) << "thread " << thread_id_ << " completed.";
        //  //  thread_condition.notify_all();
        //  //  calc_timer.Pause();
        //  //  VLOG(0) << "read_time: " << reader_timer.ElapsedUS();
        //  //  VLOG(0) << "calc_time: " << calc_timer.ElapsedUS();
        //  //  return;
        //  //}
        //  if (i == 0) {
        //    std::unique_lock<std::mutex> lk(thread_mutex);
        //    batch_id_ += 1;
        //    thread_condition.notify_all();
        //  }
        //}
        // forward pass:
        // if (i == 0) {
        //  VLOG(3) << "called notify all";
        //  std::unique_lock<std::mutex> lk(thread_mutex);
        //  thread_condition.notify_all();
        //}
        try {
          // int op_idx = 0;
          for (auto& op : ops_) {
            int op_role =
                boost::get<int>(op->Attr<int>(std::string("op_role")));
            // We run op with op_role = kLRSched only for the first macrobatch
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
              // timeline.Start();
              op->Run(*macrobatch_scopes_[i], place_);
              // if (gc) {
              //  DeleteUnusedTensors(*macrobatch_scopes_[i], op.get(),
              //                      unused_vars_, gc.get());
              //}
              // timeline.Pause();
              // auto time = timeline.ElapsedUS();
              // op_total_time[op_idx] += time;
              // if (time > op_max_time[op_idx]) {
              //  op_max_time[op_idx] = time;
              //}
              // if (time < op_min_time[op_idx]) {
              //  op_min_time[op_idx] = time;
              //}
              // op_count[op_idx] += 1;
              // op_total_time[op_idx] += time;
            }
            // op_idx++;
          }
        } catch (platform::EOFException&) {
          std::unique_lock<std::mutex> lk(thread_mutex);
          threads_completed = true;
          VLOG(3) << "thread " << thread_id_ << " completed.";
          VLOG(3) << "called notify all";
          thread_condition.notify_all();
          calc_timer.Pause();
          VLOG(0) << "read_time: " << reader_timer.ElapsedUS();
          VLOG(0) << "calc_time: " << calc_timer.ElapsedUS();
          VLOG(0) << "EOF encountered";
          // VLOG(0) << "============timeline============";
          // for (size_t i = 0; i < ops_.size(); ++i) {
          //  VLOG(0) << "op: " << op_name[i] << ", max_time: " <<
          //  op_max_time[i]
          //          << ", min_time: " << op_min_time[i]
          //          << ", mean_time: " << op_total_time[i] / op_count[i];
          //}
          // VLOG(0) << "================================";
          return;
        }
        // if (i == 0) {
        VLOG(3) << "called notify all";
        std::unique_lock<std::mutex> lk(thread_mutex);
        batch_id_ += 1;
        thread_condition.notify_all();
        //}
      }
      // backward pass
      for (int i = 0; i < num_macrobatches_; ++i) {
        // int op_idx = 0;
        for (auto& op : ops_) {
          int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
          if (op_role == static_cast<int>(OpRole::kBackward) ||
              op_role == (static_cast<int>(OpRole::kBackward) |
                          static_cast<int>(OpRole::kLoss))) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            // timeline.Start();
            op->Run(*macrobatch_scopes_[i], place_);
            // if (gc) {
            //  DeleteUnusedTensors(*macrobatch_scopes_[i], op.get(),
            //                      unused_vars_, gc.get());
            //}
            // timeline.Pause();
            // auto time = timeline.ElapsedUS();
            // op_total_time[op_idx] += time;
            // if (time > op_max_time[op_idx]) {
            //  op_max_time[op_idx] = time;
            //}
            // if (time < op_min_time[op_idx]) {
            //  op_min_time[op_idx] = time;
            //}
            // op_count[op_idx] += 1;
            // op_total_time[op_idx] += time;
          }
          // op_idx++;
        }
      }
      // update pass
      // int op_idx = 0;
      for (auto& op : ops_) {
        int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
        if (op_role == static_cast<int>(OpRole::kOptimize)) {
          VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                  << " for minibatch scope";
          // op->Run(*minibatch_scope_, place_);
          // timeline.Start();
          // op->Run(*macrobatch_scopes_[num_macrobatches_ - 1], place_);
          op->Run(*macrobatch_scopes_[num_macrobatches_ - 1], place_);
          // if (gc) {
          //  DeleteUnusedTensors(*macrobatch_scopes_[num_macrobatches_ - 1],
          //                      op.get(), unused_vars_, gc.get());
          //}
          // timeline.Pause();
          // auto time = timeline.ElapsedUS();
          // op_total_time[op_idx] += time;
          // if (time > op_max_time[op_idx]) {
          //  op_max_time[op_idx] = time;
          //}
          // if (time < op_min_time[op_idx]) {
          //  op_min_time[op_idx] = time;
          //}
          // op_count[op_idx] += 1;
          // op_total_time[op_idx] += time;
        }
        // op_idx++;
      }
      dev_ctx_->Wait();
      timeline.Pause();
      auto time = timeline.ElapsedUS();
      VLOG(0) << "batch time: " << time;
    }
  } else {
    while (true) {
      // forward pass:
      for (int i = 0; i < num_macrobatches_; ++i) {
        {
          PADDLE_ENFORCE_LE(
              local_batch_id_, batch_id_,
              "local_batch_id_ (%d) must be less than or equal to "
              "batch_id_ (%d)",
              local_batch_id_, batch_id_);
          std::unique_lock<std::mutex> lk(thread_mutex);
          if (local_batch_id_ == batch_id_ && !threads_completed) {
            thread_condition.wait(lk);
          }
          VLOG(3) << "thread " << thread_id_ << " local_batch_id_ "
                  << local_batch_id_ << " batch_id_ " << batch_id_;
          if (threads_completed) {
            VLOG(3) << "thread " << thread_id_ << " completed.";
            lk.unlock();
            calc_timer.Pause();
            VLOG(0) << "read_time: " << reader_timer.ElapsedUS();
            VLOG(0) << "calc_time: " << calc_timer.ElapsedUS();
            // VLOG(0) << "============timeline============";
            // for (size_t i = 0; i < ops_.size(); ++i) {
            //  VLOG(0) << "op: " << op_name[i] << ", max_time: " <<
            //  op_max_time[i]
            //          << ", min_time: " << op_min_time[i]
            //          << ", mean_time: " << op_total_time[i] / op_count[i];
            //}
            // VLOG(0) << "================================";
            return;
          }
          lk.unlock();
          local_batch_id_ += 1;
        }
        // int op_idx = 0;
        for (auto& op : ops_) {
          int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
          // We run op with op_role = kLRSched only for the first macrobatch
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
            // timeline.Start();
            op->Run(*macrobatch_scopes_[i], place_);
            // if (gc) {
            //  DeleteUnusedTensors(*macrobatch_scopes_[i], op.get(),
            //                      unused_vars_, gc.get());
            //}
            // timeline.Pause();
            // auto time = timeline.ElapsedUS();
            // op_total_time[op_idx] += time;
            // if (time > op_max_time[op_idx]) {
            //  op_max_time[op_idx] = time;
            //}
            // if (time < op_min_time[op_idx]) {
            //  op_min_time[op_idx] = time;
            //}
            // op_count[op_idx] += 1;
            // op_total_time[op_idx] += time;
          }
          // op_idx++;
        }
      }
      // backward pass
      for (int i = 0; i < num_macrobatches_; ++i) {
        // int op_idx = 0;
        for (auto& op : ops_) {
          int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
          if (op_role == static_cast<int>(OpRole::kBackward) ||
              op_role == (static_cast<int>(OpRole::kBackward) |
                          static_cast<int>(OpRole::kLoss))) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            // timeline.Start();
            op->Run(*macrobatch_scopes_[i], place_);
            // if (gc) {
            //  DeleteUnusedTensors(*macrobatch_scopes_[i], op.get(),
            //                      unused_vars_, gc.get());
            //}
            // timeline.Pause();
            // auto time = timeline.ElapsedUS();
            // op_total_time[op_idx] += time;
            // if (time > op_max_time[op_idx]) {
            //  op_max_time[op_idx] = time;
            //}
            // if (time < op_min_time[op_idx]) {
            //  op_min_time[op_idx] = time;
            //}
            // op_count[op_idx] += 1;
            // op_total_time[op_idx] += time;
          }
          // op_idx++;
        }
      }
      // update pass
      // int op_idx = 0;
      for (auto& op : ops_) {
        int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
        if (op_role == static_cast<int>(OpRole::kOptimize)) {
          VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                  << " for minibatch scope";
          // op->Run(*minibatch_scope_, place_);
          // timeline.Start();
          // op->Run(*macrobatch_scopes_[num_macrobatches_ - 1], place_);
          op->Run(*macrobatch_scopes_[num_macrobatches_ - 1], place_);
          // if (gc) {
          //  DeleteUnusedTensors(*macrobatch_scopes_[num_macrobatches_ - 1],
          //                      op.get(), unused_vars_, gc.get());
          //}
          timeline.Pause();
          // auto time = timeline.ElapsedUS();
          // op_total_time[op_idx] += time;
          // if (time > op_max_time[op_idx]) {
          //  op_max_time[op_idx] = time;
          //}
          // if (time < op_min_time[op_idx]) {
          //  op_min_time[op_idx] = time;
          //}
          // op_count[op_idx] += 1;
          // op_total_time[op_idx] += time;
        }
        // op_idx++;
      }
      dev_ctx_->Wait();
    }
  }
  calc_timer.Pause();
  VLOG(3) << "read_time: " << reader_timer.ElapsedUS();
  VLOG(3) << "calc_time: " << calc_timer.ElapsedUS();
}

}  // namespace framework
}  // namespace paddle
