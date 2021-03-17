/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include "paddle/fluid/pybind/fleet_py.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/communicator_common.h"
#include "paddle/fluid/distributed/fleet.h"
#include "paddle/fluid/distributed/service/communicator.h"
#include "paddle/fluid/distributed/service/env.h"
#include "paddle/fluid/distributed/service/heter_client.h"
#include "paddle/fluid/framework/fleet/index_wrapper.h"
#include "paddle/fluid/framework/fleet/index_sampler.h"

namespace py = pybind11;
using paddle::distributed::CommContext;
using paddle::distributed::Communicator;
using paddle::distributed::FleetWrapper;
using paddle::distributed::HeterClient;

namespace paddle {
namespace pybind {
void BindDistFleetWrapper(py::module* m) {
  py::class_<FleetWrapper, std::shared_ptr<FleetWrapper>>(*m,
                                                          "DistFleetWrapper")
      .def(py::init([]() { return FleetWrapper::GetInstance(); }))
      .def("load_sparse", &FleetWrapper::LoadSparseOnServer)
      .def("init_server", &FleetWrapper::InitServer)
      .def("run_server",
           (uint64_t (FleetWrapper::*)(void)) & FleetWrapper::RunServer)
      .def("run_server", (uint64_t (FleetWrapper::*)(          // NOLINT
                             const std::string&, uint32_t)) &  // NOLINT
                             FleetWrapper::RunServer)
      .def("init_worker", &FleetWrapper::InitWorker)
      .def("push_dense_params", &FleetWrapper::PushDenseParamSync)
      .def("pull_dense_params", &FleetWrapper::PullDenseVarsSync)
      .def("save_all_model", &FleetWrapper::SaveModel)
      .def("save_one_model", &FleetWrapper::SaveModelOneTable)
      .def("recv_and_save_model", &FleetWrapper::RecvAndSaveTable)
      .def("sparse_table_stat", &FleetWrapper::PrintTableStat)
      .def("stop_server", &FleetWrapper::StopServer)
      .def("stop_worker", &FleetWrapper::FinalizeWorker)
      .def("barrier", &FleetWrapper::BarrierWithTable)
      .def("shrink_sparse_table", &FleetWrapper::ShrinkSparseTable);
}

void BindPSHost(py::module* m) {
  py::class_<distributed::PSHost>(*m, "PSHost")
      .def(py::init<const std::string&, uint32_t, uint32_t>())
      .def("serialize_to_string", &distributed::PSHost::serialize_to_string)
      .def("parse_from_string", &distributed::PSHost::parse_from_string)
      .def("to_uint64", &distributed::PSHost::serialize_to_uint64)
      .def("from_uint64", &distributed::PSHost::parse_from_uint64)
      .def("to_string", &distributed::PSHost::to_string);
}

void BindCommunicatorContext(py::module* m) {
  py::class_<CommContext>(*m, "CommContext")
      .def(
          py::init<const std::string&, const std::vector<std::string>&,
                   const std::vector<std::string>&, const std::vector<int64_t>&,
                   const std::vector<std::string>&, int, bool, bool, bool, int,
                   bool>())
      .def("var_name", [](const CommContext& self) { return self.var_name; })
      .def("trainer_id",
           [](const CommContext& self) { return self.trainer_id; })
      .def("table_id", [](const CommContext& self) { return self.table_id; })
      .def("split_varnames",
           [](const CommContext& self) { return self.splited_varnames; })
      .def("split_endpoints",
           [](const CommContext& self) { return self.epmap; })
      .def("sections",
           [](const CommContext& self) { return self.height_sections; })
      .def("aggregate", [](const CommContext& self) { return self.merge_add; })
      .def("is_sparse", [](const CommContext& self) { return self.is_sparse; })
      .def("is_distributed",
           [](const CommContext& self) { return self.is_distributed; })
      .def("origin_varnames",
           [](const CommContext& self) { return self.origin_varnames; })
      .def("is_tensor_table",
           [](const CommContext& self) { return self.is_tensor_table; })
      .def("__str__", [](const CommContext& self) { return self.print(); });
}

using paddle::distributed::AsyncCommunicator;
using paddle::distributed::GeoCommunicator;
using paddle::distributed::RecvCtxMap;
using paddle::distributed::RpcCtxMap;
using paddle::distributed::SyncCommunicator;
using paddle::framework::Scope;

void BindDistCommunicator(py::module* m) {
  // Communicator is already used by nccl, change to DistCommunicator
  py::class_<Communicator, std::shared_ptr<Communicator>>(*m,
                                                          "DistCommunicator")
      .def(py::init([](const std::string& mode, const std::string& dist_desc,
                       const std::vector<std::string>& host_sign_list,
                       const RpcCtxMap& send_ctx, const RecvCtxMap& recv_ctx,
                       Scope* param_scope,
                       std::map<std::string, std::string>& envs) {
        if (mode == "ASYNC") {
          Communicator::InitInstance<AsyncCommunicator>(
              send_ctx, recv_ctx, dist_desc, host_sign_list, param_scope, envs);
        } else if (mode == "SYNC") {
          Communicator::InitInstance<SyncCommunicator>(
              send_ctx, recv_ctx, dist_desc, host_sign_list, param_scope, envs);
        } else if (mode == "GEO") {
          Communicator::InitInstance<GeoCommunicator>(
              send_ctx, recv_ctx, dist_desc, host_sign_list, param_scope, envs);
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "unsuported communicator MODE"));
        }
        return Communicator::GetInstantcePtr();
      }))
      .def("stop", &Communicator::Stop)
      .def("start", &Communicator::Start)
      .def("push_sparse_param", &Communicator::RpcSendSparseParam)
      .def("is_running", &Communicator::IsRunning)
      .def("init_params", &Communicator::InitParams);
  //  .def("recv", &Communicator::RecvNoBarrier);
}

void BindHeterClient(py::module* m) {
  py::class_<HeterClient, std::shared_ptr<HeterClient>>(*m, "HeterClient")
      .def(py::init(
          [](const std::vector<std::string>& endpoint, const int& trainer_id) {
            return HeterClient::GetInstance(endpoint, trainer_id);
          }))
      .def("stop", &HeterClient::Stop);
}

using paddle::framework::TreeIndex;
using paddle::framework::IndexWrapper;

void BindTreeIndex(py::module* m) {
  py::class_<TreeIndex>(*m, "TreeIndex")
      .def(py::init<>())
      .def("height", [](TreeIndex& self){ return self.height(); })
      .def("branch", [](TreeIndex& self){ return self.branch(); })
      .def("total_node_nums", [](TreeIndex& self) { return self.total_node_nums(); })
      .def("get_nodes_given_level", [](TreeIndex& self, int level, bool ret_code) {
           return self.get_nodes_given_level(level, ret_code);
      })
      .def("get_travel_path", [](TreeIndex& self, int id, bool ret_code, int start_level) {
           return self.get_travel_path(id, ret_code, start_level);
      });
}

void BindIndexWrapper(py::module* m) {
  py::class_<IndexWrapper, std::shared_ptr<IndexWrapper>>(*m, "IndexWrapper")
      .def(py::init(
          [](){
            return IndexWrapper::GetInstancePtr();
      }))
      .def("insert_tree_index", &IndexWrapper::insert_tree_index)
      .def("get_tree_index", &IndexWrapper::GetTreeIndex)
      .def("clear_tree", &IndexWrapper::clear_tree);
}

using paddle::framework::Sampler;
using paddle::framework::LayerWiseSampler;
using paddle::framework::BeamSearchSampler;

void BindIndexSampler(py::module* m) {
   py::class_<Sampler, std::shared_ptr<Sampler>>(*m, "Sampler")
      .def(py::init([](const std::string& mode, const std::string& name){
           if (mode == "by_layerwise") {
                return Sampler::Init<LayerWiseSampler>(name);
           } else if (mode == "by_beamsearch") {
                return Sampler::Init<BeamSearchSampler>(name);
           }
      }))
      .def("init_layerwise_conf", &Sampler::init_layerwise_conf)
      .def("init_beamsearch_conf", &Sampler::init_beamsearch_conf)
      .def("sample", &Sampler::sample);
}

}  // end namespace pybind
}  // namespace paddle
