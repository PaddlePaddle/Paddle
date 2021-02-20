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
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/service/brpc_ps_server.h"
#include "paddle/fluid/distributed/service/env.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace distributed {
class DownpourBrpcClosure;
class PSClient;
class PSServer;
}  // namespace distributed
namespace framework {
class LoDTensor;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace operators = paddle::operators;
namespace math = paddle::operators::math;
namespace memory = paddle::memory;
namespace distributed = paddle::distributed;

void CreateVarsOnScope(framework::Scope* scope, platform::CPUPlace* place) {
  auto x_var = scope->Var("x");
  x_var->GetMutable<framework::LoDTensor>();
}

void InitTensorsOnClient(framework::Scope* scope, platform::CPUPlace* place,
                         int64_t rows_numel) {
  CreateVarsOnScope(scope, place);

  auto x_var = scope->Var("x")->GetMutable<framework::LoDTensor>();
  float* x_ptr =
      x_var->mutable_data<float>(framework::DDim({1, rows_numel}), *place);
  for (int64_t i = 0; i < rows_numel; ++i) x_ptr[i] = 1.0;
}

void GetDownpourSparseTableProto(
    ::paddle::distributed::TableParameter* sparse_table_proto) {
  sparse_table_proto->set_table_id(0);
  sparse_table_proto->set_table_class("CommonSparseTable");
  sparse_table_proto->set_shard_num(256);
  sparse_table_proto->set_type(::paddle::distributed::PS_SPARSE_TABLE);
  ::paddle::distributed::TableAccessorParameter* accessor_proto =
      sparse_table_proto->mutable_accessor();
  ::paddle::distributed::CommonAccessorParameter* common_proto =
      sparse_table_proto->mutable_common();

  accessor_proto->set_accessor_class("CommMergeAccessor");
  accessor_proto->set_fea_dim(0);
  accessor_proto->set_embedx_dim(10);

  common_proto->set_name("sgd");
  common_proto->set_table_name("MergedDense");
  common_proto->set_trainer_num(1);
  common_proto->set_sync(false);
  common_proto->add_params("Param");
  common_proto->add_dims(10);
  common_proto->add_initializers("uniform_random&0&-1.0&1.0");
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

  ::paddle::distributed::TableParameter* sparse_table_proto =
      downpour_server_proto->add_downpour_table_param();
  GetDownpourSparseTableProto(sparse_table_proto);
  return server_fleet_desc;
}

::paddle::distributed::PSParameter GetWorkerProto() {
  ::paddle::distributed::PSParameter worker_fleet_desc;
  ::paddle::distributed::WorkerParameter* worker_proto =
      worker_fleet_desc.mutable_worker_param();

  ::paddle::distributed::DownpourWorkerParameter* downpour_worker_proto =
      worker_proto->mutable_downpour_worker_param();

  ::paddle::distributed::TableParameter* worker_sparse_table_proto =
      downpour_worker_proto->add_downpour_table_param();
  GetDownpourSparseTableProto(worker_sparse_table_proto);

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

  ::paddle::distributed::TableParameter* server_sparse_table_proto =
      downpour_server_proto->add_downpour_table_param();
  GetDownpourSparseTableProto(server_sparse_table_proto);

  return worker_fleet_desc;
}

/*-------------------------------------------------------------------------*/

std::string ip_ = "127.0.0.1";
uint32_t port_ = 4209;

std::vector<std::string> host_sign_list_;

std::shared_ptr<paddle::distributed::PSServer> pserver_ptr_;

std::shared_ptr<paddle::distributed::PSClient> worker_ptr_;

void RunServer() {
  ::paddle::distributed::PSParameter server_proto = GetServerProto();

  auto _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.set_ps_servers(&host_sign_list_, 1);
  pserver_ptr_ = std::shared_ptr<paddle::distributed::PSServer>(
      paddle::distributed::PSServerFactory::create(server_proto));
  std::vector<framework::ProgramDesc> empty_vec;
  framework::ProgramDesc empty_prog;
  empty_vec.push_back(empty_prog);
  pserver_ptr_->configure(server_proto, _ps_env, 0, empty_vec);
  pserver_ptr_->start(ip_, port_);
}

void RunClient(std::map<uint64_t, std::vector<paddle::distributed::Region>>&
                   dense_regions) {
  ::paddle::distributed::PSParameter worker_proto = GetWorkerProto();
  paddle::distributed::PaddlePSEnvironment _ps_env;
  auto servers_ = host_sign_list_.size();
  _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.set_ps_servers(&host_sign_list_, servers_);
  worker_ptr_ = std::shared_ptr<paddle::distributed::PSClient>(
      paddle::distributed::PSClientFactory::create(worker_proto));
  worker_ptr_->configure(worker_proto, dense_regions, _ps_env, 0);
}

void RunBrpcPushSparse() {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  auto ph_host = paddle::distributed::PSHost(ip_, port_, 0);
  host_sign_list_.push_back(ph_host.serialize_to_string());

  // Srart Server
  std::thread server_thread(RunServer);
  sleep(1);

  // Start Client
  framework::Scope client_scope;
  platform::CPUPlace place;
  InitTensorsOnClient(&client_scope, &place, 100);
  std::map<uint64_t, std::vector<paddle::distributed::Region>> dense_regions;
  dense_regions.insert(
      std::pair<uint64_t, std::vector<paddle::distributed::Region>>(0, {}));
  auto regions = dense_regions[0];
  framework::Variable* var = client_scope.FindVar("x");
  framework::LoDTensor* tensor = var->GetMutable<framework::LoDTensor>();

  RunClient(dense_regions);
  std::vector<uint64_t> fea_keys(10);
  std::vector<float> fea_values(100);
  std::vector<float> fea_temp_values(100);
  std::vector<float*> fea_value_ptr(10);
  std::vector<float*> fea_temp_value_ptr(10);

  for (size_t idx = 0; idx < fea_keys.size(); ++idx) {
    fea_keys[idx] = (uint64_t)idx;
    fea_value_ptr[idx] = fea_values.data() + idx * 10;
    fea_temp_value_ptr[idx] = fea_temp_values.data() + idx * 10;
  }

  /*-----------------------Test Server Init----------------------------------*/
  LOG(INFO) << "Run pull_sparse_param";
  auto pull_status = worker_ptr_->pull_sparse(fea_value_ptr.data(), 0,
                                              fea_keys.data(), fea_keys.size());
  pull_status.wait();
  for (size_t idx = 0; idx < tensor->numel(); ++idx) {
    fea_values.data()[idx] *= 2.0;
  }

  /*-----------------------Test Push Param----------------------------------*/

  LOG(INFO) << "Run push_sparse_param";
  paddle::distributed::DownpourBrpcClosure* closure_push_param =
      new paddle::distributed::DownpourBrpcClosure(1, [&](void* done) {
        int ret = 0;
        auto* closure = (paddle::distributed::DownpourBrpcClosure*)done;
        for (size_t i = 0; i < 1; ++i) {
          if (closure->check_response(
                  i, paddle::distributed::PS_PUSH_SPARSE_PARAM) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
      });
  auto push_status = worker_ptr_->push_sparse_param(
      0, fea_keys.data(), (const float**)fea_value_ptr.data(), fea_keys.size(),
      closure_push_param);
  push_status.wait();

  auto pull_param_status = worker_ptr_->pull_sparse(
      fea_temp_value_ptr.data(), 0, fea_keys.data(), fea_keys.size());
  pull_param_status.wait();

  for (size_t idx = 0; idx < tensor->numel(); ++idx) {
    EXPECT_FLOAT_EQ(fea_temp_values[idx], fea_values[idx]);
  }

  /*-----------------------Test Push Grad----------------------------------*/

  paddle::distributed::DownpourBrpcClosure* closure_push_grad =
      new paddle::distributed::DownpourBrpcClosure(1, [&](void* done) {
        int ret = 0;
        auto* closure = (paddle::distributed::DownpourBrpcClosure*)done;
        for (size_t i = 0; i < 1; ++i) {
          if (closure->check_response(
                  i, paddle::distributed::PS_PUSH_SPARSE_TABLE) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
      });

  LOG(INFO) << "Run pull_sparse_grad";
  std::vector<float*> push_g_vec;
  for (auto i = 0; i < static_cast<int>(fea_keys.size()); ++i) {
    push_g_vec.push_back(tensor->data<float>() + i * 10);
  }
  auto push_grad_status = worker_ptr_->push_sparse_raw_gradient(
      0, fea_keys.data(), (const float**)push_g_vec.data(), fea_keys.size(),
      closure_push_grad);
  push_grad_status.wait();

  auto pull_update_status = worker_ptr_->pull_sparse(
      fea_temp_value_ptr.data(), 0, fea_keys.data(), fea_keys.size());
  pull_update_status.wait();

  for (size_t idx = 0; idx < tensor->numel(); ++idx) {
    EXPECT_FLOAT_EQ(fea_temp_values[idx], fea_values[idx] - 1.0);
  }

  LOG(INFO) << "Run stop_server";
  worker_ptr_->stop_server();
  LOG(INFO) << "Run finalize_worker";
  worker_ptr_->finalize_worker();
  server_thread.join();
}

TEST(RunBrpcPushSparse, Run) { RunBrpcPushSparse(); }
