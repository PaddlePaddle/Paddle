/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/framework/threadpool.h"

#include <thread>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

DEFINE_int32(io_threadpool_size, 100,
             "number of threads used for doing IO, default 100");

DECLARE_int32(dist_threadpool_size);

namespace paddle {
namespace framework {
std::unique_ptr<ThreadPool> ThreadPool::threadpool_(nullptr);
std::once_flag ThreadPool::init_flag_;

ThreadPool* ThreadPool::GetInstance() {
  std::call_once(init_flag_, &ThreadPool::Init);
  return threadpool_.get();
}

void ThreadPool::Init() {
  if (threadpool_.get() == nullptr) {
    // TODO(Yancey1989): specify the max threads number
    int num_threads = std::thread::hardware_concurrency();
    if (FLAGS_dist_threadpool_size > 0) {
      num_threads = FLAGS_dist_threadpool_size;
      VLOG(1) << "set dist_threadpool_size to " << num_threads;
    }
    PADDLE_ENFORCE_GT(num_threads, 0, platform::errors::InvalidArgument(
                                          "The number of threads is 0."));
    threadpool_.reset(new ThreadPool(num_threads));
  }
}

ThreadPool::ThreadPool(int num_threads) : running_(true) {
  threads_.resize(num_threads);
  for (auto& thread : threads_) {
    // TODO(Yancey1989): binding the thread on the specify CPU number
    thread.reset(new std::thread(std::bind(&ThreadPool::TaskLoop, this)));
  }
}

ThreadPool::~ThreadPool() {
  {
    // notify all threads to stop running
    std::unique_lock<std::mutex> l(mutex_);
    running_ = false;
  }
  scheduled_.notify_all();

  for (auto& t : threads_) {
    t->join();
    t.reset(nullptr);
  }
}

void ThreadPool::TaskLoop() {
  while (true) {
    Task task;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      scheduled_.wait(
          lock, [this] { return !this->tasks_.empty() || !this->running_; });

      if (!running_ && tasks_.empty()) {
        return;
      }

      if (tasks_.empty()) {
        PADDLE_THROW(platform::errors::Unavailable(
            "Current thread has no task to Run."));
      }

      // pop a task from the task queue
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    // run the task
    task();
  }
}

std::unique_ptr<ThreadPool> ThreadPoolIO::io_threadpool_(nullptr);
std::once_flag ThreadPoolIO::io_init_flag_;

ThreadPool* ThreadPoolIO::GetInstanceIO() {
  std::call_once(io_init_flag_, &ThreadPoolIO::InitIO);
  return io_threadpool_.get();
}

void ThreadPoolIO::InitIO() {
  if (io_threadpool_.get() == nullptr) {
    // TODO(typhoonzero1986): make this configurable
    io_threadpool_.reset(new ThreadPool(FLAGS_io_threadpool_size));
  }
}

}  // namespace framework
}  // namespace paddle
