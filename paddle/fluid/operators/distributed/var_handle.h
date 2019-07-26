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

#include <time.h>
#include <condition_variable>  // NOLINT

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {
namespace distributed {

class VarHandle {
 public:
  VarHandle(const std::string ep, const std::string& method,
            const std::string& name,
            const platform::DeviceContext* p_ctx = nullptr,
            const framework::Scope* p_scope = nullptr)
      : status_(kDefaultState) {
    ep_ = ep;
    ctx_ = p_ctx;
    scope_ = p_scope;
    name_ = name;
    method_ = method;
  }

  virtual ~VarHandle() {}

 public:
  bool Wait() {
    int ret = kDefaultState;
    {
      std::unique_lock<std::mutex> lk(sync_mutex_);
      wait_cond_.wait(lk, [this] { return status_ != kDefaultState; });
      ret = status_;
    }
    VLOG(7) << "VarHandle wait:" << ret;
    return ret != kErrorState;
  }

  void Finish(bool ok) {
    {
      std::unique_lock<std::mutex> lk(sync_mutex_);
      status_ = ok ? kFinishState : kErrorState;
    }
    VLOG(7) << "VarHandle finish:" << ok;
    wait_cond_.notify_all();
  }

  std::string String() const {
    std::ostringstream s;
    s << method_ << " name:[" << name_ << "], ep:[" << ep_ << "], status:["
      << status_ << "]";
    return s.str();
  }

  std::string ep() const { return ep_; }
  const platform::DeviceContext* ctx() const { return ctx_; }
  const framework::Scope* scope() const { return scope_; }
  std::string name() const { return name_; }
  std::string method() const { return method_; }

 protected:
  // RPC endpoint.
  std::string ep_;
  const platform::DeviceContext* ctx_;
  const framework::Scope* scope_;
  // Variable name.
  std::string name_;
  // RPC method name.
  std::string method_;

 protected:
  std::mutex sync_mutex_;
  std::condition_variable wait_cond_;

  enum VarHandleStatus {
    kDefaultState = -1,
    kErrorState = 0,
    kFinishState = 1,
  };
  VarHandleStatus status_;

 private:
  DISABLE_COPY_AND_ASSIGN(VarHandle);
};

typedef std::shared_ptr<VarHandle> VarHandlePtr;

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
