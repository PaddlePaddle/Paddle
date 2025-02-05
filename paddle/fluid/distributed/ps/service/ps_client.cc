// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/ps/service/ps_client.h"

#include "glog/logging.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/service/coordinator_client.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_client.h"
#include "paddle/fluid/distributed/ps/service/ps_local_client.h"
#include "paddle/fluid/distributed/ps/table/table.h"
#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
#include "paddle/fluid/distributed/ps/service/ps_graph_client.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

namespace paddle::distributed {
REGISTER_PSCORE_CLASS(PSClient, BrpcPsClient);
REGISTER_PSCORE_CLASS(PSClient, PsLocalClient);
REGISTER_PSCORE_CLASS(PSClient, GraphBrpcClient);
REGISTER_PSCORE_CLASS(PSClient, CoordinatorClient);
#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
REGISTER_PSCORE_CLASS(PSClient, PsGraphClient);
#endif

int32_t PSClient::Configure(  // called in FleetWrapper::InitWorker
    const PSParameter &config,
    const std::map<uint64_t, std::vector<::paddle::distributed::Region>>
        &regions,
    PSEnvironment &env,
    size_t client_id) {
  _env = &env;
  _config = config;
  _dense_pull_regions = regions;
  _client_id = client_id;
  _config.mutable_worker_param()
      ->mutable_downpour_worker_param()
      ->mutable_downpour_table_param()
      ->CopyFrom(_config.server_param()
                     .downpour_server_param()
                     .downpour_table_param());

  const auto &work_param = _config.worker_param().downpour_worker_param();

  for (int i = 0; i < work_param.downpour_table_param_size(); ++i) {
    auto *accessor = CREATE_PSCORE_CLASS(
        ValueAccessor,
        work_param.downpour_table_param(i).accessor().accessor_class());
    accessor->Configure(work_param.downpour_table_param(i).accessor());
    accessor->Initialize();
    _table_accessors[work_param.downpour_table_param(i).table_id()].reset(
        accessor);
  }
  return Initialize();
}

PSClient *PSClientFactory::Create(const PSParameter &ps_config) {
  const auto &config = ps_config.server_param();
  if (!config.has_downpour_server_param()) {
    LOG(ERROR) << "miss downpour_server_param in ServerParameter";
    return nullptr;
  }

  if (!config.downpour_server_param().has_service_param()) {
    LOG(ERROR) << "miss service_param in ServerParameter.downpour_server_param";
    return nullptr;
  }

  if (!config.downpour_server_param().service_param().has_client_class()) {
    LOG(ERROR) << "miss client_class in "
                  "ServerParameter.downpour_server_param.service_param";
    return nullptr;
  }

  const auto &service_param = config.downpour_server_param().service_param();
  const auto &client_name = service_param.client_class();

  PSClient *client = nullptr;
  const auto &work_param = ps_config.worker_param().downpour_worker_param();
  bool use_gpu_graph = work_param.downpour_table_param(0).use_gpu_graph();
  VLOG(3) << "PSClient Client use_gpu_graph:" << use_gpu_graph;
  if (use_gpu_graph) {
#if defined(PADDLE_WITH_GLOO) && defined(PADDLE_WITH_HETERPS) && \
    defined(PADDLE_WITH_PSCORE)
    auto gloo = ::paddle::framework::GlooWrapper::GetInstance();
    if (client_name == "PsLocalClient" && gloo->Size() > 1) {
      client = CREATE_PSCORE_CLASS(PSClient, "PsGraphClient");
      LOG(WARNING) << "change PsLocalClient to PsGraphClient";
    } else {
      client = CREATE_PSCORE_CLASS(PSClient, client_name);
    }
#endif
  } else {
    client = CREATE_PSCORE_CLASS(PSClient, client_name);
  }
  if (client == nullptr) {
    LOG(ERROR) << "client is not registered, server_name:"
               << service_param.client_class();
    return nullptr;
  }

  TableManager::Instance().Initialize();
  VLOG(3) << "Create PSClient[" << service_param.client_class() << "] success";
  return client;
}
}  // namespace paddle::distributed
