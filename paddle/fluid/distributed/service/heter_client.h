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
#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/service/heter_serde.h"
#include "paddle/fluid/distributed/service/sendrecv.pb.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace distributed {

using MultiVarMsg = ::paddle::MultiVariableMessage;
using VarMsg = ::paddle::VariableMessage;

typedef std::function<void(void*)> HeterRpcCallbackFunc;

class OnHeterRpcDone : public google::protobuf::Closure {
 public:
  OnHeterRpcDone(HeterRpcCallbackFunc func) : handler_(func) {}
  virtual ~OnHeterRpcDone() {}
  void Run() {
    std::unique_ptr<OnHeterRpcDone> self_guard(this);
    handler_(this);
  }

  HeterRpcCallbackFunc handler_;
  MultiVariableMessage response;
  brpc::Controller cntl;
};

class HeterClient {
 public:
  virtual ~HeterClient() {}

  HeterClient() {}

  void CreateClient2XpuConnection();

  void SendAndRecvAsync(const std::string& ep,
                        const platform::DeviceContext& ctx,
                        const framework::Scope& scope,
                        const std::string& message_name,
                        const std::vector<std::string>& send_var_name,
                        const std::vector<std::string>& recv_var_name);

  // HeterClient singleton
  static std::shared_ptr<HeterClient> GetInstance(std::string endpoint) {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::distributed::HeterClient());
      std::vector<std::string> xpu_list = {endpoint};
      s_instance_->SetXpuList(xpu_list);
      s_instance_->CreateClient2XpuConnection();
    }
    return s_instance_;
  }

  std::vector<std::string>& GetXpuList() { return xpu_list_; }

  void SetXpuList(const std::vector<std::string>& xpu_list);

 private:
  // atomic<HeterClient*> s_instance_=nullptr;
  static std::shared_ptr<HeterClient> s_instance_;

 protected:
  std::vector<std::shared_ptr<brpc::Channel>> xpu_channels_;
  static bool is_initialized_;
  DISABLE_COPY_AND_ASSIGN(HeterClient);
  std::vector<std::string> xpu_list_;
};

}  // end namespace distributed
}  // end namespace paddle
