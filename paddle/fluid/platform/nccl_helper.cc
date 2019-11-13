//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/nccl_helper.h"

#include <memory>
#include <utility>

#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/string/split.h"

namespace paddle {
namespace platform {

NCCLContextMap::NCCLContextMap(const std::vector<platform::Place> &places,
                               ncclUniqueId *nccl_id, size_t num_trainers,
                               size_t trainer_id) {
  PADDLE_ENFORCE_EQ(places.empty(), false);
  order_.reserve(places.size());
  std::set<int> dev_set;
  for (auto &p : places) {
    int dev_id = boost::get<CUDAPlace>(p).device;
    order_.emplace_back(dev_id);
    dev_set.insert(dev_id);
  }
  PADDLE_ENFORCE_EQ(
      order_.size(), dev_set.size(),
      "NCCL Context Map does not support contain two or more same device");

  const int kOrders = order_.size();
  ncclComm_t comms[kOrders];
  if (num_trainers == 1 && nccl_id == nullptr) {
    LOG(INFO) << "Initialize all nccl ranks with GPU number: " << order_.size();
    std::lock_guard<std::mutex> guard(NCCLGroupGuard::NCCLMutex());
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclCommInitAll(
        comms, static_cast<int>(order_.size()), order_.data()));
  } else {
    PADDLE_ENFORCE_NOT_NULL(nccl_id);
    {
      int nranks = num_trainers * order_.size();
      NCCLGroupGuard gurad;
      for (size_t i = 0; i < order_.size(); ++i) {
        int gpu_id = order_[i];
        int rank;
        if (order_.size() > 1) {
          rank = trainer_id * order_.size() + i;
        } else {
          rank = trainer_id;
        }
        LOG(INFO) << "Initialize nccl rank:" << rank << ", nranks:" << nranks
                  << ", gpu_id:" << gpu_id;
        PADDLE_ENFORCE_CUDA_SUCCESS(cudaSetDevice(gpu_id));
        PADDLE_ENFORCE_CUDA_SUCCESS(
            dynload::ncclCommInitRank(comms + i, nranks, *nccl_id, rank));
      }
    }
  }
  int i = 0;
  for (auto &dev_id : order_) {
    auto ptr = new NCCLContext(dev_id, comms[i++]);
    contexts_.emplace(dev_id, std::unique_ptr<NCCLContext>(ptr));
  }
}

class BroadcastNCCLIdHandler : public operators::distributed::RequestHandler {
 public:
  explicit BroadcastNCCLIdHandler(bool sync_mode)
      : operators::distributed::RequestHandler(sync_mode) {}

  bool Handle(const std::string &varname, framework::Scope *scope,
              framework::Variable *var, framework::Variable **outvar,
              const int trainer_id, const std::string &out_var_name = "",
              const std::string &table_name = "") override {
    return true;
  }
};

const char NCCLReference::s_nccl_id_var_name_[] = "NCCL_ID_";

void NCCLReference::GenerateAndSend(framework::Scope *scope,
                                    const std::vector<std::string> &endpoints) {
  auto var = scope->FindVar(s_nccl_id_var_name_);
  auto id = var->GetMutable<ncclUniqueId>();
  PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGetUniqueId(id));

  operators::distributed::RPCClient *client =
      operators::distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);

  auto dev_ctx = DeviceContextPool::Instance().Get(CPUPlace());
  for (size_t i = 1; i < endpoints.size(); ++i) {
    VLOG(3) << "sending nccl id to " << endpoints[i];
    client->AsyncSendVar(endpoints[i], *dev_ctx, *scope, s_nccl_id_var_name_);
  }
  client->Wait();
  for (size_t i = 1; i < endpoints.size(); ++i) {
    client->AsyncSendBatchBarrier(endpoints[i]);
  }
  client->Wait();
  VLOG(3) << "sending completed...";
}

void NCCLReference::GetIdByServer(framework::Scope *scope,
                                  const std::vector<std::string> &endpoints) {
  BroadcastNCCLIdHandler rpc_h(true);
  std::unique_ptr<operators::distributed::RPCServer> rpc_service(
      new RPCSERVER_T(endpoints[0], 1));

  rpc_service->RegisterRPC(operators::distributed::kRequestSend, &rpc_h);
  rpc_h.SetRPCServer(rpc_service.get());

  auto dev_ctx = DeviceContextPool::Instance().Get(CPUPlace());
  rpc_h.SetScope(scope);
  rpc_h.SetDevCtx(dev_ctx);

  std::thread server_thread(std::bind(
      &operators::distributed::RPCServer::StartServer, rpc_service.get()));

  rpc_service->SetCond(operators::distributed::kRequestSend);
  VLOG(3) << "start getting nccl id from trainer 0...";
  rpc_service->WaitBarrier(operators::distributed::kRequestSend);
  VLOG(3) << "got nccl id and stop server...";
  rpc_service->ShutDown();
  VLOG(3) << "rpc server stopped";
  server_thread.join();
}

void NCCLReference::AssignNCCLId(const std::vector<std::string> &endpoints,
                                 size_t trainer_id, ncclUniqueId *nccl_id) {
  framework::Scope scope;
  auto var = scope.Var(s_nccl_id_var_name_);
  var->GetMutable<ncclUniqueId>();

  if (trainer_id == 0) {
    GenerateAndSend(&scope, endpoints);
  } else {
    GetIdByServer(&scope, endpoints);
  }
  *nccl_id = *var->GetMutable<ncclUniqueId>();
}

void NCCLReference::InitFlattenRing(const std::vector<Place> &places,
                                    const std::vector<std::string> &endpoints,
                                    size_t trainer_id, size_t nrings) {
  ncclUniqueId id;
  for (size_t i = 0; i < nrings; ++i) {
    AssignNCCLId(endpoints, trainer_id, &id);
    InitNCCLContexts(places, &id, endpoints.size(), trainer_id, &flat_rings_);
    VLOG(3) << "NCCL flatten ring " << i << " has been created";
  }
}

std::string ParseIP(const std::string &endpoint) {
  auto elems = string::Split(endpoint, ':');
  PADDLE_ENFORCE_EQ(elems.size(), 2,
                    "The expected format of endpoint is \"ip:port\" but get %s",
                    endpoint.c_str());
  return elems[0];
}

// User should keep the offset of the current endpoint in the endpoints
// equal to the trainer_id
void NCCLReference::Init2DRing(const std::vector<Place> &places,
                               const std::vector<std::string> &endpoints,
                               size_t trainer_id, size_t nrings) {
  ncclUniqueId id;
  if (places.size() == 1) {  // multi-process mode
    auto ip = ParseIP(endpoints[trainer_id]);
    std::vector<std::string> inter_eps;
    for (auto &ep : endpoints) {
      if (ep.find(ip) != std::string::npos) {
        inter_eps.push_back(ep);
      }
    }

    d2_inter_ranks_ = inter_eps.size();
    d2_exter_ranks_ = endpoints.size() / d2_inter_ranks_;

    size_t inter_rank = trainer_id % d2_inter_ranks_;
    std::vector<std::string> exter_eps;
    for (size_t i = inter_rank; i < endpoints.size(); i += d2_inter_ranks_) {
      exter_eps.push_back(endpoints[i]);
    }

    size_t exter_rank = trainer_id / d2_inter_ranks_;
    for (size_t i = 0; i < nrings; ++i) {
      AssignNCCLId(inter_eps, inter_rank, &id);
      InitNCCLContexts(places, &id, endpoints.size(), trainer_id,
                       &d2_inter_rings_);

      AssignNCCLId(exter_eps, exter_rank, &id);
      InitNCCLContexts(places, &id, endpoints.size(), trainer_id,
                       &d2_exter_rings_);

      VLOG(1) << "NCCL 2D ring " << i
              << " of multi-process mode has been created";
    }
  } else {  // multi-node with multi-threading mode
    d2_inter_ranks_ = places.size();
    d2_exter_ranks_ = endpoints.size();

    for (size_t i = 0; i < nrings; ++i) {
      InitAllNCCLContexts(places, &d2_inter_rings_);
      for (auto &place : places) {
        AssignNCCLId(endpoints, trainer_id, &id);
        InitNCCLContexts({place}, &id, endpoints.size(), trainer_id,
                         &d2_exter_rings_);
        VLOG(1) << "NCCL 2D ring " << i
                << " of multi-threading mode has been created";
      }
    }
  }

  VLOG(1) << "2D ring with 2D ranks [" << d2_exter_ranks_ << ", "
          << d2_inter_ranks_ << "] is initialized.";
}

void NCCLReference::InitFlattenRing(const std::vector<Place> &places,
                                    const std::vector<ncclUniqueId *> &nccl_ids,
                                    size_t trainer_num, size_t trainer_id) {
  if (nccl_ids.size() == 0) {
    InitAllNCCLContexts(places, &flat_rings_);
    VLOG(1) << "init local trainer";
  }

  for (size_t i = 0; i < nccl_ids.size(); ++i) {
    InitNCCLContexts(places, nccl_ids[i], trainer_num, trainer_id,
                     &flat_rings_);
    VLOG(1) << "NCCL flatten ring " << i << " has been created";
  }
}

void NCCLReference::InitHierarchicalRing(
    const std::vector<Place> &places,
    const std::vector<ncclUniqueId *> &inter_nccl_ids,
    const std::vector<ncclUniqueId *> &exter_nccl_ids, size_t nranks,
    size_t rank_id, size_t inter_ranks) {
  PADDLE_ENFORCE_GT(inter_ranks, 1, "inter_ranks:%llu must > 1", inter_ranks);

  h_inter_ranks_ = inter_ranks;
  h_exter_ranks_ = nranks / inter_ranks;

  size_t inter_rank = rank_id % inter_ranks;
  for (size_t i = 0; i < inter_nccl_ids.size(); ++i) {
    InitNCCLContexts(places, inter_nccl_ids[i], h_inter_ranks_, inter_rank,
                     &h_inter_rings_);
    VLOG(1) << "init inter_rank:" << inter_rank << ", comm no:" << i;
  }

  if (rank_id % inter_ranks == 0) {
    size_t exter_rank = rank_id / inter_ranks;
    for (size_t i = 0; i < exter_nccl_ids.size(); ++i) {
      InitNCCLContexts(places, exter_nccl_ids[i], h_exter_ranks_, exter_rank,
                       &h_exter_rings_);
      VLOG(1) << "init exter_rank:" << exter_rank << ", comm no:" << i;
    }
  }
}

void NCCLReference::AllReduce(const void *send, void *recv, size_t count,
                              ncclDataType_t datatype, ncclRedOp_t op,
                              const Place &place, cudaStream_t stream,
                              size_t order) {
  auto nccl_ctx = flat_rings_[order % flat_rings_.size()]->at(place);
  auto comm = nccl_ctx->comm();
  if (stream == nullptr) {
    stream = nccl_ctx->stream();
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(
      dynload::ncclAllReduce(send, recv, count, datatype, op, comm, stream));
}

void NCCLReference::AllReduce2D(const void *send, void *recv, size_t count,
                                ncclDataType_t datatype, ncclRedOp_t op,
                                const Place &place, cudaStream_t stream,
                                size_t order) {
  if (count % d2_inter_ranks_ != 0) {
    VLOG(0) << "The 2d internal ranks " << d2_inter_ranks_
            << " is an aliquant part of the data count " << count
            << ", and the ring-based allreduce will be used instead of 2D "
               "allreduce";
    AllReduce(send, recv, count, datatype, op, place, stream, order);
    return;
  }

  auto inter_ctx = d2_inter_rings_[order % d2_inter_rings_.size()]->at(place);
  auto inter_comm = inter_ctx->comm();
  auto exter_ctx = d2_exter_rings_[order % d2_exter_rings_.size()]->at(place);
  auto exter_comm = exter_ctx->comm();

  if (stream == nullptr) {
    // We use the stream of internal NCCLContext to keep the nccl op order.
    stream = inter_ctx->stream();
  }
  size_t part_size = count / d2_inter_ranks_;

  PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclReduceScatter(
      send, recv, part_size, datatype, op, inter_comm, stream));
  PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclAllReduce(
      recv, recv, part_size, datatype, op, exter_comm, stream));
  PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclAllGather(
      recv, recv, part_size, datatype, inter_comm, stream));
}

void NCCLReference::InitNCCLContexts(
    const std::vector<Place> &places, ncclUniqueId *nccl_id, size_t ntrainers,
    size_t trainer_id, std::vector<std::unique_ptr<NCCLContextMap>> *rings) {
  PADDLE_ENFORCE_GE(ntrainers, 1);
  PADDLE_ENFORCE_GE(trainer_id, 0);
  PADDLE_ENFORCE_LT(trainer_id, ntrainers);

  if (rings == nullptr) {
    rings = &flat_rings_;
  }
  auto ptr = new NCCLContextMap(places, nccl_id, ntrainers, trainer_id);
  rings->push_back(std::unique_ptr<NCCLContextMap>(ptr));

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLReference::Instance().ReleaseNCCLResource(); });
  });
}

void NCCLReference::ReleaseNCCLResource() {
  // CUDADeviceContext maintain the lifetime of nccl_comm_t, so we should not
  // destroy nccl_comm_t explicitly. Please refer to
  // platform::CUDADeviceContext::~CUDADeviceContext()
  for (auto &ring : flat_rings_) {
    ring.reset();
  }
  for (auto &ring : d2_inter_rings_) {
    ring.reset();
  }
  for (auto &ring : d2_exter_rings_) {
    ring.reset();
  }
  for (auto &ring : h_inter_rings_) {
    ring.reset();
  }
  for (auto &ring : h_exter_rings_) {
    ring.reset();
  }
}

}  // namespace platform
}  // namespace paddle

#endif
