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

#include <unistd.h>

#include <string>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_server.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace distributed {
class DownpourBrpcClosure;
class PSClient;
class PSServer;
}  // namespace distributed
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace phi {
class DenseTensor;
}  // namespace phi

namespace framework = paddle::framework;
namespace platform = paddle::platform;

void CreateVarsOnScope(framework::Scope* scope, phi::CPUPlace* place) {
  auto x_var = scope->Var("x");
  x_var->GetMutable<phi::DenseTensor>();
}

void InitTensorsOnClient(framework::Scope* scope,
                         phi::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);

  auto x_var = scope->Var("x")->GetMutable<phi::DenseTensor>();
  float* x_ptr = x_var->mutable_data<float>(phi::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i)
    x_ptr[i] = static_cast<float>(1.0) * static_cast<float>(i);
}

void GetDownpourDenseTableProto(
    ::paddle::distributed::TableParameter* dense_table_proto) {
  dense_table_proto->set_table_id(0);
  dense_table_proto->set_table_class("MemoryDenseTable");
  dense_table_proto->set_shard_num(256);
  dense_table_proto->set_type(::paddle::distributed::PS_DENSE_TABLE);
  ::paddle::distributed::TableAccessorParameter* accessor_proto =
      dense_table_proto->mutable_accessor();
  ::paddle::distributed::CommonAccessorParameter* common_proto =
      dense_table_proto->mutable_common();

  accessor_proto->set_accessor_class("CommMergeAccessor");
  accessor_proto->set_fea_dim(100);
  accessor_proto->set_embedx_dim(1);

  common_proto->set_name("sgd");
  common_proto->set_table_name("MergedDense");
  common_proto->set_trainer_num(1);
  common_proto->set_sync(false);
  common_proto->add_params("Param");
  common_proto->add_dims(100);
  common_proto->add_initializers("fill_constant&1.0");
  common_proto->add_params("LearningRate");
  common_proto->add_dims(1);
  common_proto->add_initializers("fill_constant&1.0");
}

::paddle::distributed::PSParameter GetServerProto() {
  // Generate server proto desc
  ::paddle::distributed::PSParameter server_fleet_desc;
  ::paddle::distributed::ServerParameter* server_proto =
      server_fleet_desc.mutable_server_param();
  ::paddle::distributed::DownpourServerParameter* downpour_server_proto =
      server_proto->mutable_downpour_server_param();
  ::paddle::distributed::ServerServiceParameter* server_service_proto =
      downpour_server_proto->mutable_service_param();
  server_service_proto->set_service_class("BrpcPsService");
  server_service_proto->set_server_class("BrpcPsServer");
  server_service_proto->set_client_class("BrpcPsClient");
  server_service_proto->set_start_server_port(0);
  server_service_proto->set_server_thread_num(12);

  ::paddle::distributed::TableParameter* dense_table_proto =
      downpour_server_proto->add_downpour_table_param();
  GetDownpourDenseTableProto(dense_table_proto);
  return server_fleet_desc;
}

::paddle::distributed::PSParameter GetWorkerProto() {
  ::paddle::distributed::PSParameter worker_fleet_desc;
  ::paddle::distributed::WorkerParameter* worker_proto =
      worker_fleet_desc.mutable_worker_param();

  ::paddle::distributed::DownpourWorkerParameter* downpour_worker_proto =
      worker_proto->mutable_downpour_worker_param();

  ::paddle::distributed::TableParameter* worker_dense_table_proto =
      downpour_worker_proto->add_downpour_table_param();
  GetDownpourDenseTableProto(worker_dense_table_proto);

  ::paddle::distributed::ServerParameter* server_proto =
      worker_fleet_desc.mutable_server_param();
  ::paddle::distributed::DownpourServerParameter* downpour_server_proto =
      server_proto->mutable_downpour_server_param();
  ::paddle::distributed::ServerServiceParameter* server_service_proto =
      downpour_server_proto->mutable_service_param();
  server_service_proto->set_service_class("BrpcPsService");
  server_service_proto->set_server_class("BrpcPsServer");
  server_service_proto->set_client_class("BrpcPsClient");
  server_service_proto->set_start_server_port(0);
  server_service_proto->set_server_thread_num(12);

  ::paddle::distributed::TableParameter* server_dense_table_proto =
      downpour_server_proto->add_downpour_table_param();
  GetDownpourDenseTableProto(server_dense_table_proto);

  return worker_fleet_desc;
}

/*-------------------------------------------------------------------------*/

const char* ip_ = "127.0.0.1";
uint32_t port_ = 4214;

std::vector<std::string> host_sign_list_;

std::shared_ptr<paddle::distributed::PSServer> pserver_ptr_;

std::shared_ptr<paddle::distributed::PSClient> worker_ptr_;

void RunServer() {
  ::paddle::distributed::PSParameter server_proto = GetServerProto();

  auto _ps_env = paddle::distributed::PaddlePSEnvironment();
  LOG(INFO) << "RUN set_ps_servers";
  _ps_env.SetPsServers(&host_sign_list_, 1);
  pserver_ptr_ = std::shared_ptr<paddle::distributed::PSServer>(
      paddle::distributed::PSServerFactory::Create(server_proto));
  LOG(INFO) << "RUN configure";
  std::vector<framework::ProgramDesc> empty_vec;
  framework::ProgramDesc empty_prog;
  empty_vec.push_back(empty_prog);
  pserver_ptr_->Configure(server_proto, _ps_env, 0, empty_vec);
  LOG(INFO) << "RUN start";
  pserver_ptr_->Start(ip_, port_);
  LOG(INFO) << "End start";
}

void RunClient(std::map<uint64_t, std::vector<paddle::distributed::Region>>&
                   dense_regions) {
  ::paddle::distributed::PSParameter worker_proto = GetWorkerProto();
  paddle::distributed::PaddlePSEnvironment _ps_env;
  auto servers_ = host_sign_list_.size();
  _ps_env = paddle::distributed::PaddlePSEnvironment();
  LOG(INFO) << "Run set_ps_servers";
  _ps_env.SetPsServers(&host_sign_list_, servers_);
  LOG(INFO) << "Run Create PSClient";
  worker_ptr_ = std::shared_ptr<paddle::distributed::PSClient>(
      paddle::distributed::PSClientFactory::Create(worker_proto));
  LOG(INFO) << "Run configure";
  worker_ptr_->Configure(worker_proto, dense_regions, _ps_env, 0);
}

void RunBrpcPushDense() {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  auto ph_host = paddle::distributed::PSHost(ip_, port_, 0);
  host_sign_list_.push_back(ph_host.SerializeToString());

  // Start Server
  std::thread server_thread(RunServer);
  sleep(1);

  // Start Client
  LOG(INFO) << "Run InitTensorsOnClient";
  framework::Scope client_scope;
  phi::CPUPlace place;
  InitTensorsOnClient(&client_scope, &place, 100);
  std::map<uint64_t, std::vector<paddle::distributed::Region>> dense_regions;
  dense_regions.insert(
      std::pair<uint64_t, std::vector<paddle::distributed::Region>>(0, {}));
  auto regions = dense_regions[0];
  framework::Variable* var = client_scope.FindVar("x");
  phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
  float* w = tensor->data<float>();
  paddle::distributed::Region reg(w, tensor->numel());
  regions.emplace_back(std::move(reg));

  LOG(INFO) << "Run RunClient";
  RunClient(dense_regions);

  /*-----------------------Test Server Init----------------------------------*/
  LOG(INFO) << "Run pull_dense_param";
  float* temp = new float[tensor->numel()]();
  std::vector<paddle::distributed::Region> temp_region;
  paddle::distributed::Region temp_reg(temp, tensor->numel());
  temp_region.emplace_back(std::move(temp_reg));
  auto pull_status =
      worker_ptr_->PullDense(temp_region.data(), temp_region.size(), 0);
  pull_status.wait();

  for (int64_t idx = 0; idx < tensor->numel(); ++idx) {
    EXPECT_FLOAT_EQ(temp[idx], 1.0);
  }

  /*-----------------------Test Push Param----------------------------------*/

  LOG(INFO) << "Run push_dense_param";
  auto push_status =
      worker_ptr_->PushDenseParam(regions.data(), regions.size(), 0);
  push_status.wait();

  pull_status = worker_ptr_->PullDense(regions.data(), regions.size(), 0);
  pull_status.wait();

  for (int64_t idx = 0; idx < tensor->numel(); ++idx) {
    EXPECT_FLOAT_EQ(w[idx], static_cast<float>(idx));
  }

  /*-----------------------Test Push Grad----------------------------------*/

  paddle::distributed::DownpourBrpcClosure* closure =
      new paddle::distributed::DownpourBrpcClosure(1, [&](void* done) {
        int ret = 0;
        auto* closure = (paddle::distributed::DownpourBrpcClosure*)done;
        for (size_t i = 0; i < 1; ++i) {
          if (closure->check_response(
                  i, paddle::distributed::PS_PUSH_DENSE_TABLE) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
      });

  LOG(INFO) << "Run pull_dense_grad";
  auto push_grad_status =
      worker_ptr_->PushDenseRawGradient(0, temp, tensor->numel(), closure);
  push_grad_status.wait();

  auto pull_update_status =
      worker_ptr_->PullDense(regions.data(), regions.size(), 0);
  pull_update_status.wait();

  for (int64_t idx = 0; idx < tensor->numel(); ++idx) {
    EXPECT_FLOAT_EQ(w[idx], static_cast<float>(idx) - 1.0);
  }

  LOG(INFO) << "Run stop_server";
  worker_ptr_->StopServer();
  LOG(INFO) << "Run finalize_worker";
  worker_ptr_->FinalizeWorker();
  server_thread.join();
}

TEST(RunBrpcPushDense, Run) { RunBrpcPushDense(); }
