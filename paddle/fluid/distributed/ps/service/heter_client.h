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
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/service/brpc_utils.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace distributed {

using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;
using VarMsg = ::paddle::distributed::VariableMessage;

typedef std::function<void(void*)> HeterRpcCallbackFunc;

class OnHeterRpcDone : public google::protobuf::Closure {
 public:
  explicit OnHeterRpcDone(HeterRpcCallbackFunc func) : handler_(func) {}
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

  HeterClient() {
    running_ = true;
    main_thread_.reset(
        new std::thread(std::bind(&HeterClient::MainThread, this)));
  }

  void CreateClient2XpuConnection();

  void SendAndRecvAsync(const platform::DeviceContext& ctx,
                        const framework::Scope& scope,
                        const std::string& message_name,
                        const std::vector<std::string>& send_var_name,
                        const std::vector<std::string>& recv_var_name,
                        const std::string& mode = "forward");

  // HeterClient singleton
  static std::shared_ptr<HeterClient> GetInstance(
      const std::vector<std::string>& endpoint,
      const std::vector<std::string>& previous_endpoint,
      const int& trainer_id) {
    if (NULL == s_instance_) {
      is_initialized_ = true;
      s_instance_.reset(new paddle::distributed::HeterClient());
      s_instance_->SetXpuList(endpoint);
      s_instance_->SetPreviousXpuList(previous_endpoint);
      s_instance_->SetTrainerID(trainer_id);
      s_instance_->CreateClient2XpuConnection();
    }
    return s_instance_;
  }

  void Stop();

  void FinalizeWorker();

  void MainThread();

  void RpcProfilerControl();

  std::future<int32_t> SendCmd(uint32_t table_id, int cmd_id,
                               const std::vector<std::string>& params);

  std::future<int32_t> StartProfiler();

  std::future<int32_t> StopProfiler();
  std::future<int32_t> StopHeterWorker();

  std::vector<std::string>& GetXpuList() { return xpu_list_; }

  void SetXpuList(const std::vector<std::string>& xpu_list) {
    xpu_list_ = xpu_list;
  }

  void SetPreviousXpuList(const std::vector<std::string>& xpu_list) {
    previous_xpu_list_ = xpu_list;
  }

  void SetTrainerID(const int& trainer_id) { trainer_id_ = trainer_id; }

 private:
  static std::shared_ptr<HeterClient> s_instance_;
  static bool is_initialized_;
  std::unique_ptr<std::thread> main_thread_{nullptr};
  std::vector<std::shared_ptr<brpc::Channel>> xpu_channels_;
  std::vector<std::shared_ptr<brpc::Channel>> previous_xpu_channels_;

  DISABLE_COPY_AND_ASSIGN(HeterClient);
  std::vector<std::string> xpu_list_;
  std::vector<std::string> previous_xpu_list_;

  bool running_ = false;
  int trainer_id_;
  bool do_server_profiler_ = false;
};

}  // end namespace distributed
}  // end namespace paddle
