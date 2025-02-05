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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/index_dataset/index_sampler.h"
#include "paddle/fluid/distributed/index_dataset/index_wrapper.h"
#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#include "paddle/fluid/distributed/ps/service/communicator/communicator_common.h"
#include "paddle/fluid/distributed/ps/service/env.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_client.h"
#include "paddle/fluid/distributed/ps/service/heter_client.h"
#include "paddle/fluid/distributed/ps/service/ps_service/graph_py_service.h"
#include "paddle/fluid/distributed/ps/wrapper/fleet.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#include "paddle/fluid/pybind/fleet_py.h"

namespace py = pybind11;
using paddle::distributed::CommContext;
using paddle::distributed::Communicator;
using paddle::distributed::FeatureNode;
using paddle::distributed::FleetWrapper;
using paddle::distributed::GraphNode;
using paddle::distributed::GraphPyClient;
using paddle::distributed::GraphPyServer;
using paddle::distributed::GraphPyService;
using paddle::distributed::HeterClient;

namespace paddle::pybind {
void BindDistFleetWrapper(py::module* m) {
  py::class_<FleetWrapper, std::shared_ptr<FleetWrapper>>(*m,
                                                          "DistFleetWrapper")
      .def(py::init([]() { return FleetWrapper::GetInstance(); }))
      .def("load_sparse", &FleetWrapper::LoadSparseOnServer)
      .def("load_model", &FleetWrapper::LoadModel)
      .def("load_one_table", &FleetWrapper::LoadModelOneTable)
      .def("init_server", &FleetWrapper::InitServer)
      .def("run_server", &FleetWrapper::RunServer)
      .def("init_worker", &FleetWrapper::InitWorker)
      .def("push_dense_params", &FleetWrapper::PushDenseParamSync)
      .def("pull_dense_params", &FleetWrapper::PullDenseVarsSync)
      .def("save_all_model", &FleetWrapper::SaveModel)
      .def("save_one_model", &FleetWrapper::SaveModelOneTable)
      .def("recv_and_save_model", &FleetWrapper::RecvAndSaveTable)
      .def("sparse_table_stat", &FleetWrapper::PrintTableStat)
      .def("save_cache_table", &FleetWrapper::SaveCacheTable)
      .def("stop_server", &FleetWrapper::StopServer)
      .def("stop_worker", &FleetWrapper::FinalizeWorker)
      .def("barrier", &FleetWrapper::BarrierWithTable)
      .def("shrink_sparse_table", &FleetWrapper::ShrinkSparseTable)
      .def("set_clients", &FleetWrapper::SetClients)
      .def("get_client_info", &FleetWrapper::GetClientsInfo)
      .def("create_client2client_connection",
           &FleetWrapper::CreateClient2ClientConnection)
      .def("client_flush", &FleetWrapper::ClientFlush)
      .def("get_cache_threshold", &FleetWrapper::GetCacheThreshold)
      .def("cache_shuffle", &FleetWrapper::CacheShuffle)
      .def("save_cache", &FleetWrapper::SaveCache)
      .def("init_fl_worker", &FleetWrapper::InitFlWorker)
      .def("push_fl_client_info_sync", &FleetWrapper::PushFLClientInfoSync)
      .def("pull_fl_strategy", &FleetWrapper::PullFlStrategy)
      .def("revert", &FleetWrapper::Revert)
      .def("set_date", &FleetWrapper::SetDate)
      .def("print_table_stat", &FleetWrapper::PrintTableStat)
      .def("check_save_pre_patch_done", &FleetWrapper::CheckSavePrePatchDone);
}

void BindPSHost(py::module* m) {
  py::class_<distributed::PSHost>(*m, "PSHost")
      .def(py::init<const std::string&, uint32_t, uint32_t>())
      .def("serialize_to_string", &distributed::PSHost::SerializeToString)
      .def("parse_from_string", &distributed::PSHost::ParseFromString)
      .def("to_uint64", &distributed::PSHost::SerializeToUint64)
      .def("from_uint64", &distributed::PSHost::ParseFromUint64)
      .def("to_string", &distributed::PSHost::ToString);
}

void BindCommunicatorContext(py::module* m) {
  py::class_<CommContext>(*m, "CommContext")
      .def(py::init<const std::string&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<int64_t>&,
                    const std::vector<std::string>&,
                    int,
                    bool,
                    bool,
                    bool,
                    int,
                    bool,
                    bool,
                    int64_t,
                    const std::vector<int32_t>&>())
      .def("var_name", [](const CommContext& self) { return self.var_name; })
      .def("remote_sparse_ids",
           [](const CommContext& self) { return self.remote_sparse_ids; })
      .def("trainer_id",
           [](const CommContext& self) { return self.trainer_id; })
      .def("table_id", [](const CommContext& self) { return self.table_id; })
      .def("program_id",
           [](const CommContext& self) { return self.program_id; })
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
      .def("is_datanorm_table",
           [](const CommContext& self) { return self.is_datanorm_table; })
      .def("__str__", [](const CommContext& self) { return self.print(); });
}

using paddle::distributed::AsyncCommunicator;
using paddle::distributed::FLCommunicator;
using paddle::distributed::GeoCommunicator;
using paddle::distributed::RecvCtxMap;
using paddle::distributed::RpcCtxMap;
using paddle::distributed::SyncCommunicator;
using paddle::framework::Scope;

void BindDistCommunicator(py::module* m) {
  // Communicator is already used by nccl, change to DistCommunicator
  py::class_<Communicator, std::shared_ptr<Communicator>>(*m,
                                                          "DistCommunicator")
      .def(py::init([](const std::string& mode,
                       const std::string& dist_desc,
                       const std::vector<std::string>& host_sign_list,
                       const RpcCtxMap& send_ctx,
                       const RecvCtxMap& recv_ctx,
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
        } else if (mode == "WITH_COORDINATOR") {
          Communicator::InitInstance<FLCommunicator>(
              send_ctx, recv_ctx, dist_desc, host_sign_list, param_scope, envs);
        } else {
          PADDLE_THROW(
              common::errors::InvalidArgument("unsupported communicator MODE"));
        }
        return Communicator::GetInstancePtr();
      }))
      .def("stop", &Communicator::Stop)
      .def("start", &Communicator::Start)
      .def("push_sparse_param", &Communicator::RpcSendSparseParam)
      .def("is_running", &Communicator::IsRunning)
      .def("init_params", &Communicator::InitParams)
      .def("pull_dense", &Communicator::PullDense)
      .def("create_client_to_client_connection",
           &Communicator::CreateC2CConnection)
      .def("get_client_info", &Communicator::GetClientInfo)
      .def("set_clients", &Communicator::SetClients)
      .def("start_coordinator", &Communicator::StartCoordinator)
      .def("query_fl_clients_info", &Communicator::QueryFLClientsInfo)
      .def("save_fl_strategy", &Communicator::SaveFLStrategy);
}

void BindHeterClient(py::module* m) {
  py::class_<HeterClient, std::shared_ptr<HeterClient>>(*m, "HeterClient")
      .def(py::init([](const std::vector<std::string>& endpoints,
                       const std::vector<std::string>& previous_endpoints,
                       const int& trainer_id) {
        return HeterClient::GetInstance(
            endpoints, previous_endpoints, trainer_id);
      }))
      .def("stop", &HeterClient::Stop);
}

void BindGraphNode(py::module* m) {
  py::class_<GraphNode>(*m, "GraphNode")
      .def(py::init<>())
      .def("get_id", &GraphNode::get_py_id)
      .def("get_feature", &GraphNode::get_feature);
}
void BindGraphPyFeatureNode(py::module* m) {
  py::class_<FeatureNode>(*m, "FeatureNode")
      .def(py::init<>())
      .def("get_id", &GraphNode::get_py_id)
      .def("get_feature", &GraphNode::get_feature);
}

void BindGraphPyService(py::module* m) {
  py::class_<GraphPyService>(*m, "GraphPyService").def(py::init<>());
}

void BindGraphPyServer(py::module* m) {
  py::class_<GraphPyServer>(*m, "GraphPyServer")
      .def(py::init<>())
      .def("start_server", &GraphPyServer::start_server)
      .def("set_up", &GraphPyServer::set_up)
      .def("add_table_feat_conf", &GraphPyServer::add_table_feat_conf);
}
void BindGraphPyClient(py::module* m) {
  py::class_<GraphPyClient>(*m, "GraphPyClient")
      .def(py::init<>())
      .def("load_edge_file", &GraphPyClient::load_edge_file)
      .def("load_node_file", &GraphPyClient::load_node_file)
      .def("set_up", &GraphPyClient::set_up)
      .def("add_table_feat_conf", &GraphPyClient::add_table_feat_conf)
      .def("pull_graph_list", &GraphPyClient::pull_graph_list)
      .def("start_client", &GraphPyClient::start_client)
      .def("batch_sample_neighbors", &GraphPyClient::batch_sample_neighbors)
      // .def("use_neighbors_sample_cache",
      //      &GraphPyClient::use_neighbors_sample_cache)
      .def("remove_graph_node", &GraphPyClient::remove_graph_node)
      .def("random_sample_nodes", &GraphPyClient::random_sample_nodes)
      .def("stop_server", &GraphPyClient::StopServer)
      .def("get_node_feat",
           [](GraphPyClient& self,
              std::string node_type,
              std::vector<int64_t> node_ids,
              std::vector<std::string> feature_names) {
             auto feats =
                 self.get_node_feat(node_type, node_ids, feature_names);
             std::vector<std::vector<py::bytes>> bytes_feats(feats.size());
             for (size_t i = 0; i < feats.size(); ++i) {
               for (size_t j = 0; j < feats[i].size(); ++j) {
                 bytes_feats[i].push_back(py::bytes(feats[i][j]));
               }
             }
             return bytes_feats;
           })
      .def("set_node_feat",
           [](GraphPyClient& self,
              std::string node_type,
              std::vector<int64_t> node_ids,
              std::vector<std::string> feature_names,
              std::vector<std::vector<py::bytes>> bytes_feats) {
             std::vector<std::vector<std::string>> feats(bytes_feats.size());
             for (size_t i = 0; i < bytes_feats.size(); ++i) {
               for (size_t j = 0; j < bytes_feats[i].size(); ++j) {
                 feats[i].push_back(std::string(bytes_feats[i][j]));
               }
             }
             self.set_node_feat(node_type, node_ids, feature_names, feats);
             return;
           })
      .def("bind_local_server", &GraphPyClient::bind_local_server);
}

using paddle::distributed::IndexNode;
using paddle::distributed::IndexWrapper;
using paddle::distributed::TreeIndex;
#ifdef PADDLE_WITH_HETERPS
using paddle::framework::GraphGpuWrapper;
using paddle::framework::NeighborSampleQuery;
using paddle::framework::NeighborSampleResult;
using paddle::framework::NodeQueryResult;
#endif

void BindIndexNode(py::module* m) {
  py::class_<IndexNode>(*m, "IndexNode")
      .def(py::init<>())
      .def("id", [](IndexNode& self) { return self.id(); })
      .def("is_leaf", [](IndexNode& self) { return self.is_leaf(); })
      .def("probability", [](IndexNode& self) { return self.probability(); });
}

void BindTreeIndex(py::module* m) {
  py::class_<TreeIndex, std::shared_ptr<TreeIndex>>(*m, "TreeIndex")
      .def(py::init([](const std::string name, const std::string path) {
        auto index_wrapper = IndexWrapper::GetInstancePtr();
        index_wrapper->insert_tree_index(name, path);
        return index_wrapper->get_tree_index(name);
      }))
      .def("height", [](TreeIndex& self) { return self.Height(); })
      .def("branch", [](TreeIndex& self) { return self.Branch(); })
      .def("total_node_nums",
           [](TreeIndex& self) { return self.TotalNodeNums(); })
      .def("emb_size", [](TreeIndex& self) { return self.EmbSize(); })
      .def("get_all_leaves",
           [](TreeIndex& self) { return self.GetAllLeaves(); })
      .def("get_nodes",
           [](TreeIndex& self, const std::vector<uint64_t>& codes) {
             return self.GetNodes(codes);
           })
      .def("get_layer_codes",
           [](TreeIndex& self, int level) { return self.GetLayerCodes(level); })
      .def("get_ancestor_codes",
           [](TreeIndex& self, const std::vector<uint64_t>& ids, int level) {
             return self.GetAncestorCodes(ids, level);
           })
      .def("get_children_codes",
           [](TreeIndex& self, uint64_t ancestor, int level) {
             return self.GetChildrenCodes(ancestor, level);
           })
      .def("get_travel_codes",
           [](TreeIndex& self, uint64_t id, int start_level) {
             return self.GetTravelCodes(id, start_level);
           });
}

void BindIndexWrapper(py::module* m) {
  py::class_<IndexWrapper, std::shared_ptr<IndexWrapper>>(*m, "IndexWrapper")
      .def(py::init([]() { return IndexWrapper::GetInstancePtr(); }))
      .def("insert_tree_index", &IndexWrapper::insert_tree_index)
      .def("get_tree_index", &IndexWrapper::get_tree_index)
      .def("clear_tree", &IndexWrapper::clear_tree);
}

#ifdef PADDLE_WITH_HETERPS
void BindNodeQueryResult(py::module* m) {
  py::class_<NodeQueryResult>(*m, "NodeQueryResult")
      .def(py::init<>())
      .def("initialize", &NodeQueryResult::initialize)
      .def("display", &NodeQueryResult::display)
      .def("get_val", &NodeQueryResult::get_val)
      .def("get_len", &NodeQueryResult::get_len);
}
void BindNeighborSampleQuery(py::module* m) {
  py::class_<NeighborSampleQuery>(*m, "NeighborSampleQuery")
      .def(py::init<>())
      .def("initialize", &NeighborSampleQuery::initialize)
      .def("display", &NeighborSampleQuery::display);
}

void BindNeighborSampleResult(py::module* m) {
  py::class_<NeighborSampleResult>(*m, "NeighborSampleResult")
      .def(py::init<>())
      .def("initialize", &NeighborSampleResult::initialize)
      .def("get_len", &NeighborSampleResult::get_len)
      .def("get_val", &NeighborSampleResult::get_actual_val)
      .def("get_sampled_graph", &NeighborSampleResult::get_sampled_graph)
      .def("display", &NeighborSampleResult::display);
}

void BindGraphGpuWrapper(py::module* m) {
  py::class_<GraphGpuWrapper, std::shared_ptr<GraphGpuWrapper>>(
      *m, "GraphGpuWrapper")
      .def(py::init([]() { return GraphGpuWrapper::GetInstance(); }))
      .def("neighbor_sample", &GraphGpuWrapper::graph_neighbor_sample_v3)
      .def("graph_neighbor_sample",
           py::overload_cast<int, uint64_t*, int, int>(
               &GraphGpuWrapper::graph_neighbor_sample))
      .def("graph_neighbor_sample",
           py::overload_cast<int, int, std::vector<uint64_t>&, int>(
               &GraphGpuWrapper::graph_neighbor_sample))
      .def("set_device", &GraphGpuWrapper::set_device)
      .def("set_feature_separator", &GraphGpuWrapper::set_feature_separator)
      .def("set_infer_mode", &GraphGpuWrapper::set_infer_mode)
      .def("set_slot_feature_separator",
           &GraphGpuWrapper::set_slot_feature_separator)
      .def("init_service", &GraphGpuWrapper::init_service)
      .def("set_up_types", &GraphGpuWrapper::set_up_types)
      .def("query_node_list", &GraphGpuWrapper::query_node_list)
      .def("add_table_feat_conf", &GraphGpuWrapper::add_table_feat_conf)
      .def("load_edge_file",
           py::overload_cast<std::string, std::string, bool>(
               &GraphGpuWrapper::load_edge_file))
      .def("load_edge_file",
           py::overload_cast<std::string,
                             std::string,
                             int,
                             bool,
                             const std::vector<bool>&,
                             bool>(&GraphGpuWrapper::load_edge_file))
      .def("load_node_and_edge", &GraphGpuWrapper::load_node_and_edge)
      .def("calc_edge_type_limit", &GraphGpuWrapper::calc_edge_type_limit)
      .def("show_mem", &GraphGpuWrapper::show_mem)
      .def("upload_batch",
           py::overload_cast<int, int, const std::string&>(
               &GraphGpuWrapper::upload_batch))
      .def(
          "upload_batch",
          py::overload_cast<int, int, int, int>(&GraphGpuWrapper::upload_batch))
      .def(
          "get_all_id",
          py::overload_cast<int, int, int, std::vector<std::vector<uint64_t>>*>(
              &GraphGpuWrapper::get_all_id))
      .def("get_all_id",
           py::overload_cast<int, int, std::vector<std::vector<uint64_t>>*>(
               &GraphGpuWrapper::get_all_id))
      .def("init_metapath", &GraphGpuWrapper::init_metapath)
      .def("get_node_type_size", &GraphGpuWrapper::get_node_type_size)
      .def("get_edge_type_size", &GraphGpuWrapper::get_edge_type_size)
      .def("clear_metapath_state", &GraphGpuWrapper::clear_metapath_state)
      .def("load_next_partition", &GraphGpuWrapper::load_next_partition)
      .def("make_partitions", &GraphGpuWrapper::make_partitions)
      .def("make_complementary_graph",
           &GraphGpuWrapper::make_complementary_graph)
      .def("set_search_level", &GraphGpuWrapper::set_search_level)
      .def("init_search_level", &GraphGpuWrapper::init_search_level)
      .def("get_partition_num", &GraphGpuWrapper::get_partition_num)
      .def("get_partition", &GraphGpuWrapper::get_partition)
      .def("load_node_weight", &GraphGpuWrapper::load_node_weight)
      .def("export_partition_files", &GraphGpuWrapper::export_partition_files)
      .def("load_node_file",
           py::overload_cast<std::string, std::string>(
               &GraphGpuWrapper::load_node_file))
      .def("load_node_file",
           py::overload_cast<std::string, std::string, int>(
               &GraphGpuWrapper::load_node_file))
      .def("release_graph", &GraphGpuWrapper::release_graph)
      .def("release_graph_edge", &GraphGpuWrapper::release_graph_edge)
      .def("release_graph_node", &GraphGpuWrapper::release_graph_node)
      .def("finalize", &GraphGpuWrapper::finalize)
      .def("set_node_iter_from_file",
           py::overload_cast<std::string, std::string, int, bool, bool>(
               &GraphGpuWrapper::set_node_iter_from_file))
      .def("set_node_iter_from_graph",
           py::overload_cast<bool, bool>(
               &GraphGpuWrapper::set_node_iter_from_graph));
}
#endif

using paddle::distributed::IndexSampler;
using paddle::distributed::LayerWiseSampler;

void BindIndexSampler(py::module* m) {
  py::class_<IndexSampler, std::shared_ptr<IndexSampler>>(*m, "IndexSampler")
      .def(py::init([](const std::string& mode, const std::string& name) {
        if (mode == "by_layerwise") {
          return IndexSampler::Init<LayerWiseSampler>(name);
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "Unsupported IndexSampler Type!"));
        }
      }))
      .def("init_layerwise_conf", &IndexSampler::init_layerwise_conf)
      .def("init_beamsearch_conf", &IndexSampler::init_beamsearch_conf)
      .def("sample", &IndexSampler::sample);
}
}  // namespace paddle::pybind
