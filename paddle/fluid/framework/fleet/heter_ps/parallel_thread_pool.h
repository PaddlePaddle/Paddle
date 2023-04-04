/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <mutex>
#include <atomic>
#include <pthread.h>
#include <condition_variable>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/framework/fleet/heter_ps/thread_barrier.h"

namespace paddle {
namespace framework {

class ParallelThreadPool {
public:
  ParallelThreadPool(int thread_num = 0) {
    need_exit_ = false;
    work_func_ = nullptr;
    task_finished_.store(true);
    waiting_task_begin_cnt_.store(0);
    running_thread_cnt_.store(0);
   
    if (thread_num > 0) {
      init(thread_num);
    }
  }

  ~ParallelThreadPool() {
    if (running_thread_cnt_.load() == 0) {
      return;
    }

    cv_mtx_.lock();
    need_exit_ = true;
    cv_mtx_.unlock();

    wait_task();

    while (waiting_task_begin_cnt_.load() > 0) {
      task_begin_cv_.notify_all();
    }

    for (size_t i = 0; i < threads_.size(); i++) {
      threads_[i].join();
    }
  }

  void init(int thread_num) {
    thread_num_ = thread_num;    
    barrier_.reset(thread_num);
    threads_.resize(thread_num);
    for (int i = 0; i < thread_num; i++) {
      threads_[i] = std::thread([this](int tid) {
        this->running_thread_cnt_.fetch_add(1);
        std::unique_lock<std::mutex> lock(this->cv_mtx_);
        while (true) {
          if (this->need_exit_) {
            break;
          }
          this->waiting_task_begin_cnt_.fetch_add(1);
          this->task_begin_cv_.wait(lock);
          this->waiting_task_begin_cnt_.fetch_sub(1);

          if (this->work_func_ != nullptr) {
            lock.unlock();
            this->work_func_(tid);
            this->barrier_.wait();

            lock.lock(); 
            if (tid == 0 ) {
              task_finished_.store(true);    
              this->task_end_cv_.notify_all();
            }
          } else {
            continue;
          }
        }
        this->running_thread_cnt_.fetch_sub(1);
      }, i);
    }
    while (this->running_thread_cnt_.load() < thread_num) {
      continue;
    }
  }

  int get_thread_num() {
    return thread_num_;
  }

  void set_task(std::function<void(int)> func) {
    PADDLE_ENFORCE_EQ(task_finished_.load(), true);
    PADDLE_ENFORCE_EQ(running_thread_cnt_.load(), thread_num_);
    PADDLE_ENFORCE_EQ(waiting_task_begin_cnt_.load(), thread_num_);
    std::unique_lock<std::mutex> lock(cv_mtx_);
    work_func_ = func;
    task_finished_.store(false);
    task_begin_cv_.notify_all();
  }

  void wait_task() {
    std::unique_lock<std::mutex> lock(cv_mtx_);
    if (task_finished_.load()) {
      return;
    }
    task_end_cv_.wait(lock);
    work_func_ = nullptr;
  }

private:
  int thread_num_;
  bool need_exit_ = false;

  std::mutex cv_mtx_;
  ThreadBarrier barrier_;
  std::atomic<int> running_thread_cnt_;
  std::atomic<int> waiting_task_begin_cnt_;
  std::atomic<bool> task_finished_;
  std::vector<std::thread> threads_;
  std::condition_variable task_begin_cv_;
  std::condition_variable task_end_cv_;
  std::function<void(int)> work_func_ = nullptr;
};


}  // end namespace framework
}  // end namespace paddle
