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

namespace paddle {
namespace distributed {

PD_DECLARE_int32(pserver_timeout_ms);
PD_DECLARE_int32(pserver_connect_timeout_ms);
PD_DECLARE_uint64(total_fl_client_size);
PD_DECLARE_uint32(coordinator_wait_all_clients_max_time);

using CoordinatorServiceFunc =
    std::function<int32_t(const CoordinatorReqMessage& request,
                          CoordinatorResMessage* response,
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

  void SaveFLClientInfo(const CoordinatorReqMessage& request) {
    auto client_id = request.client_id();
    const std::string& str_params = request.str_params();
    // each client is allowed to send empty message to maintain heartbeat(i.e.
    // use staleness msg)
    std::unique_lock<std::mutex> lck(_mtx);
    if (str_params.size() != 0) {
      _client_info_mp[client_id] = str_params;
    } else {
      LOG(INFO) << "fl-ps > content in request from " << client_id
                << " is null";
    }
    fl_client_ids.insert(client_id);
    _fl_clients_count++;
    // TODO(ziyoujiyi): how to process when a client loss connection?
    if (_fl_clients_count.load() == last_round_total_fl_clients_num) {
      _is_all_clients_info_collected = true;
      _cv.notify_one();
    }
    lck.unlock();
    VLOG(0) << "last_round_total_fl_clients_num: "
            << last_round_total_fl_clients_num
            << ", has received fl client num: " << _fl_clients_count.load();
    return;
  }

  std::unordered_map<uint32_t, std::string> QueryFLClientsInfo() {
    platform::Timer timeline;
    double query_wait_time = 0.0;
    timeline.Start();
    auto f = [&]() -> bool {
      while (query_wait_time <
             FLAGS_coordinator_wait_all_clients_max_time) {  // in case that
                                                             // some
                                                             // clients down
        if (_is_all_clients_info_collected == true) {
          // LOG(INFO) << "fl-ps > _is_all_clients_info_collected";
          return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        timeline.Pause();
        query_wait_time += timeline.ElapsedSec();
      }
      // LOG(WARNING) << "fl-ps > query_wait_time exceed!";
      return true;
    };

    std::unique_lock<std::mutex> lck(_mtx);
    _cv.wait(lck, f);
    lck.unlock();

    _is_all_clients_info_collected = false;
    _fl_clients_count.store(0);
    return _client_info_mp;
  }

 public:
  std::unordered_map<uint32_t, std::string> _client_info_mp;
  std::set<uint32_t> fl_client_ids;
  uint32_t last_round_total_fl_clients_num = 0;
  bool _is_all_clients_info_collected = false;

 private:
  std::mutex _mtx;
  std::condition_variable _cv;
  std::atomic<uint32_t> _fl_clients_count{0};
};

class CoordinatorService : public PsService {
 public:
  CoordinatorService() {
    _coordinator_service_handle = std::make_shared<CoordinatorServiceHandle>();
  }

  virtual ~CoordinatorService() {}

  virtual void Initialize() {
    _service_handle_map[PUSH_FL_CLIENT_INFO_SYNC] =
        std::bind(&CoordinatorService::SaveFLClientInfo,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3);
  }

  virtual void FLService(::google::protobuf::RpcController* controller,
                         const CoordinatorReqMessage* request,
                         CoordinatorResMessage* response,
                         ::google::protobuf::Closure* done);

  int32_t SaveFLClientInfo(const CoordinatorReqMessage& request,
                           CoordinatorResMessage* response UNUSED,
                           brpc::Controller* cntl UNUSED) {
    _coordinator_service_handle->SaveFLClientInfo(request);
    return 0;
  }

  void SetTotalFLClientsNum(uint32_t all_fl_clients_num) {
    if (_coordinator_service_handle.get() != nullptr) {
      _coordinator_service_handle->last_round_total_fl_clients_num =
          all_fl_clients_num;
    } else {
      LOG(ERROR) << "fl-ps > _coordinator_service_handle is null in "
                    "CoordinatorService";
    }
    return;
  }

  std::set<uint32_t> GetFLClientIds() {
    return _coordinator_service_handle->fl_client_ids;
  }

  std::unordered_map<uint32_t, std::string> QueryFLClientsInfo() {
    return _coordinator_service_handle->QueryFLClientsInfo();
  }

 private:
  std::shared_ptr<CoordinatorServiceHandle> _coordinator_service_handle;
  std::unordered_map<int32_t, CoordinatorServiceFunc> _service_handle_map;
  std::mutex _mtx;
};

class CoordinatorClient : public BrpcPsClient {
 public:
  CoordinatorClient() : _coordinator_id(0) {}

  virtual ~CoordinatorClient() {}

  using BrpcPsClient::Initialize;
  int32_t Initialize(const std::vector<std::string>& trainer_endpoints);

  void SetTotalFLClientsNum(uint32_t all_fl_clients_num) {
    _service.SetTotalFLClientsNum(all_fl_clients_num);
    this->_total_clients_num = all_fl_clients_num;
    return;
  }

  int32_t StartClientService();

  void SaveFLStrategy(
      const std::unordered_map<uint32_t, std::string>& fl_strategy) {
    for (auto it = fl_strategy.begin(); it != fl_strategy.end(); it++) {
      uint32_t client_id = it->first;
      _fl_strategy_mp[client_id] = it->second;
    }
    std::unique_lock<std::mutex> lck(_mtx);
    _is_fl_strategy_ready = true;
    _cv.notify_all();
    return;
  }

  void WaitForFLStrategyReady() {
    std::unique_lock<std::mutex> lck(_mtx);
    _cv.wait(lck, [=]() { return _is_fl_strategy_ready; });
  }

  void SendFLStrategy(const uint32_t& client_id);

  void ResetFLStrategyFlag() { _is_fl_strategy_ready = false; }

  void SetDefaultFLStrategy() {
    for (size_t i = 0; i < _total_clients_num; i++) {
      _fl_strategy_mp[i] = "";
    }
    return;
  }

  std::set<uint32_t> GetFLClientIds() { return _service.GetFLClientIds(); }

  std::unordered_map<uint32_t, std::string> QueryFLClientsInfo() {
    return _service.QueryFLClientsInfo();
  }

  void SetEndpoint(const std::string& endpoint) {
    _endpoint = std::move(endpoint);
  }

 public:
  size_t _coordinator_id;
  uint32_t _total_clients_num;
  std::string _endpoint;
  std::vector<std::array<std::shared_ptr<brpc::Channel>, 1>>
      _pserver_channels;  // coordinator2pserver
  std::unordered_map<uint32_t, std::shared_ptr<brpc::Channel>>
      _fl_client_channels;  // coordinator2psclient
  brpc::Server _server;
  CoordinatorService _service;
  std::unordered_map<uint32_t, std::string> _fl_strategy_mp;
  bool _is_fl_strategy_ready = false;
  std::mutex _mtx;
  std::condition_variable _cv;
};

}  // namespace distributed
}  // namespace paddle
