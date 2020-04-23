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

#include <memory>
#include <atomic>
#include <ctime>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/heter_service.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {

typedef std::function<void(void*)> HeterRpcCallbackFunc;

class OnHeterRpcDone: public google::protobuf::Closure {
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
  
  HeterWrapper() {
  }
  
  static void HeterRpcCallBack(HeterResponse* response, brpc::Controller* cntl, HeterCpuWorker* worker, std::shared_ptr<HeterTask> task);

  void CreateClient2XpuConnection();
  
  void RegisterServiceHandler(HeterServiceHandler func);

  void StartXpuService(const std::string& ip, uint32_t port);
  
  void CallRemoteXpu(std::shared_ptr<HeterTask> task, HeterCpuWorker* worker);
  
  void SerializeToReq(const std::string& varname, Scope* scope, VariableMessage* req_var);

  framework::proto::VarType::Type ToVarType(VariableMessage::Type type);
  
  void DeSerializeToTensor(Scope* scope, const VariableMessage& req_var);
  
  // HeterWrapper singleton
  static std::shared_ptr<HeterWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::HeterWrapper());
    }
    return s_instance_;
  }
  
  std::vector<std::string>& GetXpuList() {
    return xpu_list_;
  }

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
