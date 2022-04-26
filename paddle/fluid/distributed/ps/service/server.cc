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

#include "paddle/fluid/distributed/ps/service/server.h"

#include "glog/logging.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_server.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_server.h"
#include "paddle/fluid/distributed/ps/service/ps_local_server.h"
#include "paddle/fluid/distributed/ps/table/table.h"

namespace paddle {
namespace distributed {

REGISTER_PSCORE_CLASS(PSServer, BrpcPsServer);
REGISTER_PSCORE_CLASS(PSServer, PsLocalServer);
REGISTER_PSCORE_CLASS(PsBaseService, BrpcPsService);
REGISTER_PSCORE_CLASS(PSServer, GraphBrpcServer);
REGISTER_PSCORE_CLASS(PsBaseService, GraphBrpcService);

PSServer *PSServerFactory::Create(const PSParameter &ps_config) {
  const auto &config = ps_config.server_param();

  if (!config.has_downpour_server_param()) {
    LOG(ERROR) << "miss downpour_server_param in ServerParameter";
    return NULL;
  }

  if (!config.downpour_server_param().has_service_param()) {
    LOG(ERROR) << "miss service_param in ServerParameter.downpour_server_param";
    return NULL;
  }

  if (!config.downpour_server_param().service_param().has_server_class()) {
    LOG(ERROR) << "miss server_class in "
                  "ServerParameter.downpour_server_param.service_param";
    return NULL;
  }

  const auto &service_param = config.downpour_server_param().service_param();
  PSServer *server =
      CREATE_PSCORE_CLASS(PSServer, service_param.server_class());
  if (server == NULL) {
    LOG(ERROR) << "server is not registered, server_name:"
               << service_param.server_class();
    return NULL;
  }
  TableManager::Instance().Initialize();
  return server;
}

int32_t PSServer::Configure(
    const PSParameter &config, PSEnvironment &env, size_t server_rank,
    const std::vector<framework::ProgramDesc> &server_sub_program) {
  scope_.reset(new framework::Scope());
  _config = config.server_param();
  _rank = server_rank;
  _environment = &env;
  _shuffled_ins =
      paddle::framework::MakeChannel<std::pair<uint64_t, std::string>>();
  size_t shard_num = env.GetPsServers().size();

  const auto &downpour_param = _config.downpour_server_param();

  uint32_t barrier_table = UINT32_MAX;
  uint32_t global_step_table = UINT32_MAX;

  for (size_t i = 0; i < downpour_param.downpour_table_param_size(); ++i) {
    auto *table = CREATE_PSCORE_CLASS(
        Table, downpour_param.downpour_table_param(i).table_class());

    if (downpour_param.downpour_table_param(i).table_class() ==
        "BarrierTable") {
      barrier_table = downpour_param.downpour_table_param(i).table_id();
    }
    if (downpour_param.downpour_table_param(i).table_class() ==
        "GlobalStepTable") {
      global_step_table = downpour_param.downpour_table_param(i).table_id();
    }

    table->SetProgramEnv(scope_.get(), place_, &server_sub_program);
    table->SetShard(_rank, shard_num);
    table->Initialize(downpour_param.downpour_table_param(i),
                      config.fs_client_param());
    _table_map[downpour_param.downpour_table_param(i).table_id()].reset(table);
  }

  if (barrier_table != UINT32_MAX) {
    _table_map[barrier_table]->SetTableMap(&_table_map);
  }
  if (global_step_table != UINT32_MAX) {
    _table_map[global_step_table]->SetTableMap(&_table_map);
  }

  return Initialize();
}
}  // namespace distributed
}  // namespace paddle
