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

#include <string>
#include <vector>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/framework/data_feed.pb.h"

#include "paddle/fluid/pybind/fleet_wrapper_py.h"

namespace py = pybind11;

namespace paddle::pybind {
void BindFleetWrapper(py::module* m) {
  py::class_<framework::FleetWrapper, std::shared_ptr<framework::FleetWrapper>>(
      *m, "Fleet")
      .def(py::init([]() { return framework::FleetWrapper::GetInstance(); }))
      .def("push_dense", &framework::FleetWrapper::PushDenseVarsSync)
      .def("pull_dense", &framework::FleetWrapper::PullDenseVarsSync)
      .def("init_server", &framework::FleetWrapper::InitServer)
      .def("run_server",
           (uint64_t(framework::FleetWrapper::*)(void)) &
               framework::FleetWrapper::RunServer)
      .def("run_server",
           (uint64_t(framework::FleetWrapper::*)(  // NOLINT
               const std::string&,
               uint32_t)) &  // NOLINT
               framework::FleetWrapper::RunServer)
      .def("init_worker", &framework::FleetWrapper::InitWorker)
      .def("init_model", &framework::FleetWrapper::PushDenseParamSync)
      .def("save_model", &framework::FleetWrapper::SaveModel)
      .def("get_cache_threshold", &framework::FleetWrapper::GetCacheThreshold)
      .def("cache_shuffle", &framework::FleetWrapper::CacheShuffle)
      .def("save_cache", &framework::FleetWrapper::SaveCache)
      .def("save_multi_table_one_path",
           &framework::FleetWrapper::SaveMultiTableOnePath)
      .def("save_model_with_whitelist",
           &framework::FleetWrapper::SaveWithWhitelist)
      .def("load_model", &framework::FleetWrapper::LoadModel)
      .def("load_table_with_whitelist",
           &framework::FleetWrapper::LoadWithWhitelist)
      .def("clear_model", &framework::FleetWrapper::ClearModel)
      .def("clear_one_table", &framework::FleetWrapper::ClearOneTable)
      .def("stop_server", &framework::FleetWrapper::StopServer)
      .def("finalize_worker", &framework::FleetWrapper::FinalizeWorker)
      .def("gather_servers", &framework::FleetWrapper::GatherServers)
      .def("gather_clients", &framework::FleetWrapper::GatherClients)
      .def("get_clients_info", &framework::FleetWrapper::GetClientsInfo)
      .def("create_client2client_connection",
           &framework::FleetWrapper::CreateClient2ClientConnection)
      .def("shrink_sparse_table", &framework::FleetWrapper::ShrinkSparseTable)
      .def("shrink_dense_table", &framework::FleetWrapper::ShrinkDenseTable)
      .def("print_table_stat", &framework::FleetWrapper::PrintTableStat)
      .def("set_file_num_one_shard",
           &framework::FleetWrapper::SetFileNumOneShard)
      .def("client_flush", &framework::FleetWrapper::ClientFlush)
      .def("load_from_paddle_model",
           &framework::FleetWrapper::LoadFromPaddleModel)
      .def("load_model_one_table", &framework::FleetWrapper::LoadModelOneTable)
      .def("set_client2client_config",
           &framework::FleetWrapper::SetClient2ClientConfig)
      .def("set_pull_local_thread_num",
           &framework::FleetWrapper::SetPullLocalThreadNum)
      .def("confirm", &framework::FleetWrapper::Confirm)
      .def("revert", &framework::FleetWrapper::Revert)
      .def("save_model_one_table", &framework::FleetWrapper::SaveModelOneTable)
      .def("save_model_one_table_with_prefix",
           &framework::FleetWrapper::SaveModelOneTablePrefix)
      .def("set_date", &framework::FleetWrapper::SetDate)
      .def("copy_table", &framework::FleetWrapper::CopyTable)
      .def("copy_table_by_feasign",
           &framework::FleetWrapper::CopyTableByFeasign);
}  // end FleetWrapper
}  // namespace paddle::pybind
