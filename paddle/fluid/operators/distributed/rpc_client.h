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

#pragma once

#include <string>
#include "gflags/gflags.h"

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"

DECLARE_int32(rpc_deadline);

namespace paddle {
namespace operators {
namespace distributed {

class RPCHandleCls {
 public:
  RPCHandleCls();
  virtual ~RPCHandleCls();

 public:
  bool Wait() {
    {
      std::unique_lock<std::mutex> lk(sync_mutex_);
      sync_cond_.wait(lk, [] { return true; });
    }
    return ok_;
  }

  void Finish(bool ok) {
    {
      std::unique_lock<std::mutex> lk(sync_mutex_);
      ok_ = ok;
    }
    sync_cond_.notify_all();
  }

 protected:
  // mutex for Wait RPC return.
  std::mutex sync_mutex_;
  std::condition_variable sync_cond_;
  bool ok_;
};

typedef std::shared_ptr<RPCHandleCls> RPCHandle;

class RPCClient {
 public:
  RPCClient() {}
  virtual ~RPCClient() {}
  virtual RPCHandle AsyncSendVar(const std::string& ep,
                                 const platform::DeviceContext& ctx,
                                 const framework::Scope& scope,
                                 const std::string& var_name,
                                 int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual RPCHandle AsyncGetVar(const std::string& ep,
                                const platform::DeviceContext& ctx,
                                const framework::Scope& scope,
                                const std::string& var_name,
                                int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual RPCHandle AsyncPrefetchVar(const std::string& ep,
                                     const platform::DeviceContext& ctx,
                                     const framework::Scope& scope,
                                     const std::string& in_var_name,
                                     const std::string& out_var_name,
                                     int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual RPCHandle AsyncSendBatchBarrier(
      const std::string& ep, int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual RPCHandle AsyncSendFetchBarrier(
      const std::string& ep, int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual RPCHandle AsyncCheckpointNotify(
      const std::string& ep, const std::string& dir,
      int64_t time_out = FLAGS_rpc_deadline) = 0;

  virtual RPCHandle AsyncSendComplete(
      const std::string& ep, int64_t time_out = FLAGS_rpc_deadline) = 0;

  // Complete tells all the pserver instances that finishe the training,
  // the pserver can reduce it's barrier count, and continue to train
  // with other trainers.
  virtual void SendComplete() = 0;

  virtual bool Wait() = 0;

  virtual bool Wait(RPCHandle h) = 0;

  template <typename T>
  static RPCClient* GetInstance() {
    std::call_once(init_flag_, &RPCClient::Init<T>);
    return rpc_client_.get();
  }

  // Init is called by GetInstance.
  template <typename T>
  static void Init() {
    if (rpc_client_.get() == nullptr) {
      rpc_client_.reset(new T());
      rpc_client_->InitImpl();
    }
  }

 protected:
  virtual void InitImpl() {}

 private:
  static std::once_flag init_flag_;
  static std::unique_ptr<RPCClient> rpc_client_;
};
}  // namespace distributed
}  // namespace operators
}  // namespace paddle
