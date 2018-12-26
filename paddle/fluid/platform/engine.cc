// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>  // NOLINT
#include <sstream>
#include <type_traits>

#include <chrono>
#include "engine.h"
#include "engine_impl.h"
#include "thread_utils.h"

namespace engine {

// ----------------------------------------------------------------------------
// Multi-thread Engine
// ----------------------------------------------------------------------------
void ThreadedResource::AppendDependency(OperationHandle opr, bool is_write) {
  DLOG(INFO) << "append " << (is_write ? " write " : " read ") << " dependency";
  {
    std::lock_guard<std::mutex> l(mut_);
    DLOG(INFO) << "t " << std::this_thread::get_id() << " enter";
    queue_.emplace_back(opr, is_write);

    ProcessQueueFront();
    DLOG(INFO) << "t " << std::this_thread::get_id() << " leave";
  }
}

void ThreadedResource::FinishedDependency(OperationHandle opr, bool is_write) {
  DLOG(INFO) << "finish dependency " << name_;
  std::lock_guard<std::mutex> l(mut_);
  {
    DLOG(INFO) << "t " << std::this_thread::get_id() << " enter";
    if (is_write) {
      write_is_running_ = false;
    } else {
      running_read_count_--;
    }

    ProcessQueueFront();
    DLOG(INFO) << "t " << std::this_thread::get_id() << " leave";
  }
  DLOG(INFO) << debug_string();
}

// NOTE Not thread safe.
void ThreadedResource::ProcessQueueFront() {
#if ENGINE_DEBUG
  if (!queue_.empty()) {
    if (queue_.front().is_write) {
      CHECK(running_read_count_.load() >= 0) << "value: "
                                             << running_read_count_;
    }
  }
#endif
  // LOG(INFO) << "process queue front";
  if (queue_.empty()) return;
  // read task is running, but the front is write, hold.
  if (running_read_count_ > 0 && queue_.front().is_write) return;
  // a write operation is running and not finished, hold.
  if (write_is_running_) return;

  // LOG(INFO) << "continue to check queue";
  if (queue_.front().is_write) {
    // Should wait for all previous read finish.
    if (running_read_count_ > 0) {
      return;
    }
    // write dependency is ready, dispatch this write operation and update
    // write_is_running_
    auto opr = queue_.front().operation;
    auto topr = opr->template Cast<ThreadedOperation>();
    {
      std::lock_guard<std::mutex> l(opr_mut);
      topr->UpdateResourceReady(1, name_);
      if (topr->ReadyToExecute()) {
        // dispatch write operation
        dispatcher_(opr);
      }
    }
    queue_.pop_front();
    write_is_running_ = true;
    CHECK_EQ(running_read_count_, 0);
    // read operation
  } else {
    while (!queue_.empty() && !queue_.front().is_write && !write_is_running_) {
      running_read_count_++;
      auto opr = queue_.front().operation->template Cast<ThreadedOperation>();

      {
        std::lock_guard<std::mutex> l(opr_mut);
        opr->UpdateResourceReady(1, name_);
        if (opr->ReadyToExecute()) {
          dispatcher_(queue_.front().operation);
        }
      }
      queue_.pop_front();
    }
  }
}

std::string ThreadedResource::debug_string() const {
  std::stringstream ss;
  if (queue_.empty()) return "";
  ss << name() << " ";
  for (auto &opr : queue_) {
    auto opr_ptr = opr.operation->Cast<ThreadedOperation>();
    ss << "\t" << opr_ptr->name << ":" << opr.is_write << " "
       << "\tpending_read: " << running_read_count_
       << "\tpending_write: " << write_is_running_ << "\n";
  }
  return ss.str();
}

class MultiThreadEngine : public Engine {
 public:
  void PushAsync(OperationHandle opr, RunContext ctx) override {
#if USE_PROFILE
    auto opr_name = opr->Cast<ThreadedOperation>()->name;
// profile::Profiler::Get()->AddPushAction(opr_name);
#endif
    inc_num_pendding_tasks();

    /*
    auto dispatcher = [this](OperationHandle opr) {
      DLOG(INFO) << "dispatch the opr";
      this->PushToExecute(opr);
    };
     */

    auto topr = opr->Cast<ThreadedOperation>();
    topr->ctx = ctx;
    for (auto res : topr->read_res) {
      VLOG(4) << "inc read " << res->template Cast<ThreadedResource>()->name();
      CHECK(res);
      res->template Cast<ThreadedResource>()->AppendDependency(opr, false);
    }
    for (auto res : topr->write_res) {
      VLOG(4) << "inc write " << res->template Cast<ThreadedResource>()->name();
      res->template Cast<ThreadedResource>()->AppendDependency(opr, true);
    }

    // LOG(INFO) << "status: \n" << StatusInfo();
  }

  void PushSync(SyncFn fn, RunContext ctx,
                const std::vector<ResourceHandle> &read_res,
                const std::vector<ResourceHandle> &write_res) override {
    AsyncFn afn = [&](RunContext ctx, CallbackOnComplete cb) { fn(ctx); };
    auto opr = NewOperation(afn, read_res, write_res);
    PushAsync(opr, ctx);
  }

  OperationHandle NewOperation(AsyncFn fn,
                               const std::vector<ResourceHandle> &read_res,
                               const std::vector<ResourceHandle> &write_res,
                               const std::string &name = "") override {
    auto opr = std::make_shared<ThreadedOperation>(this, fn, read_res,
                                                   write_res, name);
    CHECK_EQ(opr->Cast<ThreadedOperation>()->engine, (void *)this);
    return opr;
  }

  void WaitForAllFinished() override {
    // LOG(INFO) << "num_pending_tasks_ " << num_pending_tasks_;
    std::unique_lock<std::mutex> l(mut_);
    finish_cond_.wait(
        l, [this]() { return num_pending_tasks_ == 0 || terminated_; });
    DLOG(INFO) << "WaitForAllFinished done";
  }

  void WaitForResource(const std::vector<ResourceHandle> &res) override {
    AsyncFn fn;
    // TODO
  }

  // Push the opr to device and execute it.
  virtual void PushToExecute(OperationHandle opr) = 0;

  static CallbackOnComplete CreateCompleteCallback(const void *engine,
                                                   OperationHandle opr) {
    // LOG(INFO) << "call CallbackOnComplete";
    DLOG(INFO) << "create Callback engine " << engine;
    CHECK_EQ(opr->Cast<ThreadedOperation>()->engine, engine);
    static CallbackOnComplete::Fn fn = [](OperationHandle opr) {
      auto engine = opr->Cast<ThreadedOperation>()->engine;
      auto ptr = opr->Cast<ThreadedOperation>();
      for (const auto &var : ptr->read_res) {
        var->Cast<ThreadedResource>()->FinishedDependency(opr, false);
      }
      for (const auto &var : ptr->write_res) {
        var->Cast<ThreadedResource>()->FinishedDependency(opr, true);
      }
      auto engine_ptr = static_cast<MultiThreadEngine *>(engine);
      engine_ptr->dec_num_pending_tasks();
      /*
#if USE_PROFILE
      profile::Profiler::Get()->AddFinishAction(
          opr->Cast<ThreadedOperation>()->name);
#endif
       */
    };
    return CallbackOnComplete(opr, &fn, (void *)engine);
  }

 protected:
  void inc_num_pendding_tasks() {
    inc_count++;
    num_pending_tasks_++;
  }
  void dec_num_pending_tasks() {
    dec_count++;
    num_pending_tasks_--;
    DLOG(INFO) << " engine pending tasks: " << num_pending_tasks_ << " "
               << inc_count << " " << dec_count;
    finish_cond_.notify_all();
  }
  // NOTE should be updated by Terminate method.
  // Whether the engine is terminated.
  std::atomic<bool> terminated_{false};
  // number of tasks in engine.
  // NOTE should be updated in CallbackOnComplted.
  std::atomic<int> num_pending_tasks_{0};
  // Condition variable used to determine whether all the tasks are finished.
  std::condition_variable finish_cond_;
  std::atomic<uint64_t> sync_counter_;
  std::mutex mut_;
  int inc_count{0};
  int dec_count{0};
};

// ----------------------------------------------------------------------------
// Multi-thread Engine With Thread Pools
// ----------------------------------------------------------------------------
template <typename OprType>
struct ThreadQueueBlock {
  explicit ThreadQueueBlock(int n_threads)
      : workers(n_threads, [this] {
          OperationHandle o;
          while (true) {
            bool suc = task_queue.Pop(&o);
            // LOG(INFO) << "really pop one task";
            if (!suc) {
              LOG(WARNING) << "thread " << std::this_thread::get_id()
                           << " stop ...";
              return;
            }
            // DLOG(INFO) << "queue.size: " << task_queue.Size();
            auto ptr = o->Cast<OprType>();
            auto engine = ptr->engine;
            auto complete_callback =
                MultiThreadEngine::CreateCompleteCallback(engine, o);
            CHECK_EQ(complete_callback.engine, engine);
            CHECK(ptr->fn);
            // LOG(INFO) << "task " << ptr->name << " execute";
            ptr->fn(ptr->ctx, complete_callback);
          }
        }) {}

  swiftcpp::thread::TaskQueue<OperationHandle> task_queue;
  swiftcpp::thread::ThreadPool workers;
};

class MultiThreadEnginePooled final : public MultiThreadEngine {
 public:
  MultiThreadEnginePooled(int n_common_threads = 10, int n_io_threads = 1)
      : common_task_workers_(
            new ThreadQueueBlock<ThreadedOperation>(n_common_threads)),
        io_task_workers_(
            new ThreadQueueBlock<ThreadedOperation>(n_io_threads)) {}

  ~MultiThreadEnginePooled() { Terminate(); }

  ResourceHandle NewResource(const std::string &name = "") override;
  void PushToExecute(OperationHandle opr) override;
  void Terminate() override;
  std::string StatusInfo() const override;

 private:
  std::unique_ptr<ThreadQueueBlock<ThreadedOperation>> common_task_workers_;
  std::unique_ptr<ThreadQueueBlock<ThreadedOperation>> io_task_workers_;
};

ResourceHandle MultiThreadEnginePooled::NewResource(const std::string &name) {
  ThreadedResource::Dispatcher dispatcher = [this](OperationHandle opr) {
    DLOG(INFO) << "thread " << std::this_thread::get_id() << " dispatch opr "
               << opr->Cast<ThreadedOperation>()->name;
    PushToExecute(opr);
  };
  return std::make_shared<ThreadedResource>(dispatcher, name);
}

void MultiThreadEnginePooled::Terminate() {
  if (!terminated_) {
    DLOG(WARNING) << "MultiThreadEnginePooled terminated.";
    terminated_ = true;
    common_task_workers_->task_queue.SignalForKill();
    io_task_workers_->task_queue.SignalForKill();
  }
}

void MultiThreadEnginePooled::PushToExecute(OperationHandle opr) {
  auto ptr = opr->Cast<ThreadedOperation>();
  VLOG(4) << "run task " << opr->Cast<ThreadedOperation>()->name;
  VLOG(4) << "queue size " << common_task_workers_->task_queue.Size();
  if (ptr->ctx.property == kCPU_GPU_Copy ||
      ptr->ctx.property == kGPU_CPU_Copy) {
    VLOG(2) << "push to io queue";
    io_task_workers_->task_queue.Push(opr);
  } else {
    VLOG(2) << "push to common task queue";
    common_task_workers_->task_queue.Push(opr);
  }
}

std::shared_ptr<Engine> CreateEngine(const std::string &kind,
                                     EngineProperty prop) {
  if (kind == "DebugEngine") {
    return std::make_shared<DebugEngine>();
  } else if (kind == "MultiThreadEnginePooled") {
    return std::make_shared<MultiThreadEnginePooled>(
        prop.num_cpu_threads, prop.num_threads_gpu_copy_per_device);
  }
  return nullptr;
}

std::string MultiThreadEnginePooled::StatusInfo() const {
  std::stringstream ss;
  ss << "io task queue size: " << io_task_workers_->task_queue.Size() << "\n";
  ss << "io queue thread count: " << io_task_workers_->workers.Size() << "\n";
  ss << "common task queue size: " << common_task_workers_->task_queue.Size()
     << "\n";
  ss << "common queue thread count: " << common_task_workers_->workers.Size()
     << "\n";
  return ss.str();
}

};  // namespace engine
