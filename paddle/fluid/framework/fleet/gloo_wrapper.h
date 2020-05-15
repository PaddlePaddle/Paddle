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

#pragma once

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#ifdef _LINUX
#include <sys/types.h>
#include <unistd.h>
#endif
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#ifdef PADDLE_WITH_GLOO
#include <gloo/allgather.h>
#include <gloo/allreduce.h>
#include <gloo/barrier.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/tcp/device.h>
#endif
#include "paddle/fluid/framework/variable_helper.h"

namespace gloo {
namespace rendezvous {

#ifdef PADDLE_WITH_GLOO
class HdfsStore : public gloo::rendezvous::Store {
#else
class HdfsStore {
#endif
 public:  // NOLINT
  explicit HdfsStore(const std::string& path);

  virtual ~HdfsStore() {}

  virtual void set(const std::string& key, const std::vector<char>& data);

  virtual std::vector<char> get(const std::string& key);

  virtual void wait(const std::vector<std::string>& keys);

  virtual void wait(const std::vector<std::string>& keys,
                    const std::chrono::milliseconds& timeout);

  std::string EncodeName(const std::string& name);

  std::string TmpPath(const std::string& name);

  std::string ObjectPath(const std::string& name);

  bool Check(const std::vector<std::string>& keys);

  std::string path_;
  int wait_sleep_ms_;
  std::chrono::seconds wait_timeout_;
  int retry_times_;
};

}  // namespace rendezvous
}  // namespace gloo

namespace paddle {
namespace framework {

class GlooWrapper {
 public:
  GlooWrapper() {}

  virtual ~GlooWrapper() {}

  void Init(int rank, int size, const std::string& path,
            const std::string& fs_name, const std::string& fs_ugi,
            const std::string& iface, const std::string& prefix);

  int Rank() {
    CHECK_EQ(is_initialized_, true);
    return rank_;
  }

  int Size() {
    CHECK_EQ(is_initialized_, true);
    return size_;
  }

  void Barrier() {
    CHECK_EQ(is_initialized_, true);
#ifdef PADDLE_WITH_GLOO
    gloo::BarrierOptions opts(context_);
    gloo::barrier(opts);
#endif
  }

  template <typename T>
  std::vector<T> AllReduce(std::vector<T>& sendbuf,            // NOLINT
                           const std::string& mode = "sum") {  // NOLINT
    CHECK_EQ(is_initialized_, true);
    std::vector<T> recvbuf(sendbuf.size(), T());
    CHECK_EQ(sendbuf.size() == recvbuf.size(), true);
#ifdef PADDLE_WITH_GLOO
    gloo::AllreduceOptions opts(context_);
    opts.setInput(sendbuf.data(), sendbuf.size());
    opts.setOutput(recvbuf.data(), recvbuf.size());
    if (mode == "sum") {
      opts.setReduceFunction(
          static_cast<void (*)(void*, const void*, const void*, size_t)>(
              &gloo::sum<T>));
    } else if (mode == "max") {
      opts.setReduceFunction(
          static_cast<void (*)(void*, const void*, const void*, size_t)>(
              &gloo::max<T>));
    } else if (mode == "min") {
      opts.setReduceFunction(
          static_cast<void (*)(void*, const void*, const void*, size_t)>(
              &gloo::min<T>));
    } else {
      PADDLE_ENFORCE_EQ(0, 1, paddle::platform::errors::InvalidArgument(
                                  "AllReduce mode not known: " + mode));
    }
    gloo::allreduce(opts);
#endif
    return recvbuf;
  }

  template <typename T>
  std::vector<T> AllGather(T& input) {  // NOLINT
    CHECK_EQ(is_initialized_, true);
    std::vector<T> ret(size_, T());
#ifdef PADDLE_WITH_GLOO
    gloo::AllgatherOptions opts(context_);
    opts.setInput(&input, 1);
    opts.setOutput(ret.data(), size_);
    gloo::allgather(opts);
#endif
    return std::move(ret);
  }

 protected:
  bool is_initialized_ = false;
#ifdef PADDLE_WITH_GLOO
  std::shared_ptr<gloo::Context> context_ = nullptr;
#endif
  int rank_ = 0;
  int size_ = 0;
};

}  // namespace framework
}  // namespace paddle
