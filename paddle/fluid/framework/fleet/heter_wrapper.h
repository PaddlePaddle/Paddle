/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#if defined(PADDLE_WITH_PSLIB) && !defined(PADDLE_WITH_HETERPS)
#include "paddle/fluid/framework/heter_service.h"
#include "paddle/fluid/framework/heter_util.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {

class HeterCpuWorker;

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
  HeterResponse response;
  brpc::Controller cntl;
};

class HeterWrapper {
 public:
  virtual ~HeterWrapper() {
    server_.Stop(1000);
    server_.Join();
  }

  HeterWrapper() {}

  static void HeterRpcCallBack(HeterResponse* response, brpc::Controller* cntl,
                               HeterCpuWorker* worker,
                               std::shared_ptr<HeterTask> task);

  void CreateClient2XpuConnection();

  void RegisterServiceHandler(int cmd, HeterServiceHandler func);

  void StartXpuService(const std::string& ip, uint32_t port);

  void CallRemoteXpu(std::shared_ptr<HeterTask> task, HeterCpuWorker* worker,
                     int mpi_rank, std::vector<std::string>& send_vars);

  void CallRemoteXpuSync(std::shared_ptr<HeterTask> task,
                         HeterCpuWorker* worker, int mpi_rank,
                         std::vector<std::string>& send_vars);

  void StopXpuService(int num);

  void EndPass(Scope* scope, int num);

  void SerializeToReq(const std::string& varname, Scope* scope,
                      VariableMessage* req_var);

  framework::proto::VarType::Type ToVarType(VariableMessage::Type type);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void DeSerializeToTensor(Scope* scope, const VariableMessage& req_var,
                           platform::Place place, gpuStream_t stream);
#endif
  void DeSerializeToTensor(Scope* scope, const VariableMessage& req_var,
                           platform::Place place);
  // HeterWrapper singleton
  static std::shared_ptr<HeterWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::HeterWrapper());
    }
    return s_instance_;
  }

  std::vector<std::string>& GetXpuList() { return xpu_list_; }

  void SetXpuList(const std::vector<std::string>& xpu_list);

 private:
  static std::shared_ptr<HeterWrapper> s_instance_;

 protected:
  std::vector<std::shared_ptr<brpc::Channel>> xpu_channels_;
  brpc::Server server_;
  HeterXpuService service_;
  static bool is_initialized_;
  DISABLE_COPY_AND_ASSIGN(HeterWrapper);
  std::vector<std::string> xpu_list_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
