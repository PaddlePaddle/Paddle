/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/operators/nvjpeg_decoder.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

struct NvjpegDecodeWork {
  const uint8_t* bit_stream;
  size_t bit_len;
  LoDTensor* tensor;
  framework::ExecutionContext ctx;
};

class NvjpegDecoderThreadPool {
  public:
    NvjpegDecoderThreadPool(const int num_threads, const std::string mode)
      : threads_(num_threads),
        mode_(mode),
        shutdown_(false),
        running_(false),
        completed_(false),
        outstand_works_(0) {
      PADDLE_ENFORCE_GT(num_threads, 0, platform::errors::InvalidArgument(
                        "num_threads shoule be a positive interger, "
                        "but got %d", num_threads));
      for (int i = 0; i < num_threads; i++) {
        threads_.emplace_back(
            std::thread(std::bind(&NvjpegDecoderThreadPool::ThreadLoop, this, i)));
      }
    }

    void AddWork(std::shared_ptr<NvjpegDecodeWork> work) {
      work_queue_.push_back(work);
    }

    void RunAll(const bool wait, const bool sort = true) {
      // Sort images in length desending order
      if (sort) SortWorkByLengthDescend();

      {
        std::lock_guard<std::mutex> lock(mutex_);
        completed_ = false;
        running_ = true;
      }
      running_cond_.notify_all();

      if (wait) WaitTillWorksCompleted();
    }

    void WaitTillWorksCompleted() {
      std::unique_lock<std::mutex> lock(mutex_);
      completed_cond_.wait(lock, [this] { return this->completed_; });
      running_ = false;
    }

    void Shutdown() {
      std::lock_guard<std::mutex> lock(mutex_);

      running_ = false;
      shutdown_.store(true);
      running_cond_.notify_all();

      work_queue_.clear();

      for (auto &thread : threads_) {
        thread.join();
      }
    }

  private:
    std::vector<std::thread> threads_;
    std::string mode_;

    std::deque<std::shared_ptr<NvjpegDecodeWork>> work_queue_;
    std::mutex mutex_;

    std::atomic<bool> shutdown_;
    std::condition_variable running_cond_;
    bool running_;
    std::condition_variable completed_cond_;
    bool completed_;

    int outstand_works_;

    void SortWorkByLengthDescend() {
      std::lock_guard<std::mutex> lock(mutex_);
      std::sort(work_queue_.begin(), work_queue_.end(),
          [](const std::shared_ptr<NvjpegDecodeWork> a,
             const std::shared_ptr<NvjpegDecodeWork> b) {
              return b->bit_len < a->bit_len;
          });
    }

    void ThreadLoop(const int thread_idx) {
      NvjpegDecoder* decoder = new NvjpegDecoder(mode_);

      while (!shutdown_.load()) {
        std::unique_lock<std::mutex> lock(mutex_);
        running_cond_.wait(lock, [this] { return running_ && !work_queue_.empty(); });
        if (shutdown_.load()) break;

        auto work = work_queue_.front();
        work_queue_.pop_front();
        outstand_works_++;
        lock.unlock();

        decoder->Run(work->bit_stream, work->bit_len, work->tensor, work->ctx);

        lock.lock();
        outstand_works_--;
        if (outstand_works_ == 0 && work_queue_.empty()) {
          completed_ = true;
          lock.unlock();
          completed_cond_.notify_one();
        }
      }
    }
};

}  // namespace operators
}  // namespace paddle
