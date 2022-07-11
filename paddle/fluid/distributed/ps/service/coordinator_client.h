// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <ThreadPool.h>

#include <memory>
#include <string>
#include <vector>

#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/service/brpc_utils.h"
#include "paddle/fluid/distributed/ps/service/ps_client.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"

DECLARE_int32(pserver_timeout_ms);
DECLARE_int32(pserver_connect_timeout_ms);
DECLARE_uint64(total_fl_client_size);
DECLARE_uint32(coordinator_wait_all_clients_max_time);

namespace paddle {
namespace distributed {

using CoordinatorServiceFunc = std::function<int32_t(
    const CoordinatorReqMessage& request, CoordinatorResMessage* response,
    brpc::Controller* cntl)>;

class ClientReportedInfo {
 public:
  ClientReportedInfo() {}
  ~ClientReportedInfo() {}
  uint32_t client_id;
  uint32_t iteration_idx;
  double auc = 0.0;
};

class CoordinatorServiceHandle {
 public:
  CoordinatorServiceHandle() {}

  virtual ~CoordinatorServiceHandle() {}

  void SaveFlClientReportedInfo(const CoordinatorReqMessage& request) {
    auto client_id = request.client_id();
    const std::string& str_params = request.str_params();
    VLOG(0) << ">>> recved client: " << client_id << ", info: " << str_params;
    VLOG(0) << ">>> last_round_total_fl_clients_num: "
            << last_round_total_fl_clients_num;
    std::unique_lock<std::mutex> lk(mtx_);
    if (str_params.size() != 0) {
      _client_info_mp[client_id] =
          str_params;  // each client send empty message to maintain,
                       // heartbeat(i.e. use staleness msg)
    }
    fl_client_ids.insert(client_id);
    lk.unlock();
    fl_clients_count_++;
    // how to know all clients have reported params?
    // how to do when a client loss connection?
    if (fl_clients_count_.load() == last_round_total_fl_clients_num) {
      _is_all_clients_info_collected = true;
    } else {
      VLOG(0) << "total fl client num is: " << last_round_total_fl_clients_num
              << "req fl client num is: " << fl_clients_count_;
    }
    return;
  }

  std::unordered_map<uint32_t, std::string> QueryFlClientsInfo() {
    VLOG(0) << ">>> Entering QueryFlClientsInfo!";
    platform::Timer timeline;
    timeline.Start();
    double coordinator_wait_time = 0.0;
    while (coordinator_wait_time <
           FLAGS_coordinator_wait_all_clients_max_time) {  // in case that some
                                                           // clients down
      if (_is_all_clients_info_collected == true) {
        VLOG(0) << ">>> _is_all_clients_info_collected";
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      VLOG(0) << "waiting for all fl clients info collected!";
      timeline.Pause();
      coordinator_wait_time += timeline.ElapsedSec();
    }
    _is_all_clients_info_collected = false;
    fl_clients_count_.store(0);
    return _client_info_mp;
  }

  void InitDefaultFlStrategy() {
    for (size_t i = 0; i < last_round_total_fl_clients_num; i++) {
      _fl_strategy_mp[i] = "JOIN";
    }
    return;
  }

  void SaveFlStrategy(
      const std::unordered_map<uint32_t, std::string>& fl_strategy) {
    VLOG(0) << ">>> Entering SaveFlStrategy!";
    for (auto it = fl_strategy.begin(); it != fl_strategy.end(); it++) {
      uint32_t client_id = it->first;
      _fl_strategy_mp[client_id] = it->second;
    }
    _is_fl_strategy_ready = true;
    return;
  }

 public:
  std::unordered_map<uint32_t, std::string> _client_info_mp;
  std::unordered_map<uint32_t, std::string> _fl_strategy_mp;
  std::set<uint32_t> fl_client_ids;
  bool _is_fl_strategy_ready = false;
  uint32_t last_round_total_fl_clients_num = 0;
  bool _is_all_clients_info_collected = false;

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<uint32_t> fl_clients_count_{0};
};

class CoordinatorService : public PsService {
 public:
  CoordinatorService() {
    _coordinator_service_handle = std::make_shared<CoordinatorServiceHandle>();
  }

  virtual ~CoordinatorService() {}

  virtual void Initialize() {
    _service_handle_map[FL_PUSH_PARAMS_SYNC] = std::bind(
        &CoordinatorService::SaveFlClientReportedInfo, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  }

  virtual void FlService(::google::protobuf::RpcController* controller,
                         const CoordinatorReqMessage* request,
                         CoordinatorResMessage* response,
                         ::google::protobuf::Closure* done);

  int32_t SaveFlClientReportedInfo(const CoordinatorReqMessage& request,
                                   CoordinatorResMessage* response,
                                   brpc::Controller* cntl) {
    _coordinator_service_handle->SaveFlClientReportedInfo(request);
    return 0;
  }

  void InitTotalFlClientNum(uint32_t all_fl_clients_num) {
    if (_coordinator_service_handle.get() != nullptr) {
      _coordinator_service_handle->last_round_total_fl_clients_num =
          all_fl_clients_num;
    } else {
      LOG(ERROR) << "_coordinator_service_handle is null in CoordinatorService";
    }
    return;
  }

  void InitDefaultFlStrategy() {
    _coordinator_service_handle->InitDefaultFlStrategy();
  }

  void SetFlStrategyReady(bool flag) {
    _coordinator_service_handle->_is_fl_strategy_ready = flag;
    return;
  }

  bool IsFlStrategyReady() {
    return _coordinator_service_handle->_is_fl_strategy_ready;
  }

  std::set<uint32_t> GetFlClientIds() {
    return _coordinator_service_handle->fl_client_ids;
  }

  std::unordered_map<uint32_t, std::string> QueryFlClientsInfo() {
    return _coordinator_service_handle->QueryFlClientsInfo();
  }

  void SaveFlStrategy(
      const std::unordered_map<uint32_t, std::string>& fl_strategy) {
    _coordinator_service_handle->SaveFlStrategy(fl_strategy);
    return;
  }

  CoordinatorServiceHandle* GetCoordinatorServiceHandlePtr() {
    return _coordinator_service_handle.get();
  }

  void SetEndpoint(const std::string& endpoint) {}

 private:
  size_t _rank;
  PSClient* _client;
  std::shared_ptr<CoordinatorServiceHandle> _coordinator_service_handle;
  std::unordered_map<int32_t, CoordinatorServiceFunc> _service_handle_map;
  std::mutex _mtx;
};

class CoordinatorClient : public BrpcPsClient {
 public:
  CoordinatorClient() : _coordinator_id(0) {}

  virtual ~CoordinatorClient() {}

  int32_t Initialize(const std::vector<std::string>& trainer_endpoints);

  void InitTotalFlClientNum(uint32_t all_fl_clients_num) {
    _service.InitTotalFlClientNum(all_fl_clients_num);
    this->_total_client_num = all_fl_clients_num;
    return;
  }

  int32_t StartClientService();

  void SendFlStrategy(const uint32_t& client_id);

  void SetFlStrategyReady(bool flag) { _service.SetFlStrategyReady(flag); }

  bool IsFlStrategyReady() { return _service.IsFlStrategyReady(); }

  std::set<uint32_t> GetFlClientIds() { return _service.GetFlClientIds(); }

  std::unordered_map<uint32_t, std::string> QueryFlClientsInfo() {
    return _service.QueryFlClientsInfo();
  }

  void SaveFlStrategy(
      const std::unordered_map<uint32_t, std::string>& fl_strategy) {
    _service.SaveFlStrategy(fl_strategy);
    return;
  }

  void SetEndpoint(const std::string& endpoint) {
    _endpoint = std::move(endpoint);
  }

 public:
  size_t _coordinator_id;
  uint32_t _total_client_num;
  std::string _endpoint;
  std::vector<std::array<std::shared_ptr<brpc::Channel>, 1>>
      _pserver_channels;  // coordinator2pserver
  std::unordered_map<uint32_t, std::shared_ptr<brpc::Channel>>
      _fl_client_channels;  // coordinator2psclient
  brpc::Server _server;
  CoordinatorService _service;
  std::mutex _mtx;
};

}  // namespace distributed
}  // namespace paddle
