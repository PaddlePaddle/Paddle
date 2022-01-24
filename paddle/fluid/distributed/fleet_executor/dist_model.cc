// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include "paddle/fluid/distributed/fleet_executor/dist_model.h"
#include "paddle/fluid/distributed/fleet_executor/fleet_executor.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace distributed {

namespace {
bool IsPersistable(const framework::VarDesc *var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST &&
      var->GetType() != framework::proto::VarType::RAW) {
    return true;
  }
  return false;
}
}  // namespace

bool DistModel::Init() {
  /* TODO(fleet exe dev): implement this funct */
  bool init_method = (!config_.model_dir.empty() || config_.program_desc);
  PADDLE_ENFORCE_EQ(init_method, true,
                    platform::errors::InvalidArgument(
                        "One of model dir or program desc must be provided to "
                        "dist model inference."));
  if (config_.program_desc) {
    PADDLE_ENFORCE_NOT_NULL(
        config_.scope, platform::errors::InvalidArgument(
                           "Scope must be provided to dist model inference if "
                           "program desc has been provided."));
  }
  if (!PreparePlace()) {
    return false;
  }
  if (!config_.program_desc) {
    if (config_.scope) {
      LOG(WARNING) << "The provided scope will be ignored if model dir has "
                      "also been provided.";
    }
    if (!PrepareScope()) {
      return false;
    }
    if (!PrepareProgram()) {
      return false;
    }
  } else {
    program_.reset(config_.program_desc);
    scope_.reset(config_.scope);
  }
  if (!PrepareFeedAndFetch()) {
    return false;
  }
  if (!CommInit()) {
    return false;
  }
  if (!PrepareFleetExe()) {
    return false;
  }
  return true;
}

bool DistModel::PreparePlace() {
  if (config_.place == "GPU") {
    place_ = paddle::platform::CUDAPlace(config_.device_id);
  } else if (config_.place == "CPU") {
    place_ = paddle::platform::CPUPlace();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Place must be choosen from GPU or CPU, but got %s.", config_.place));
  }
  return true;
}

bool DistModel::CommInit() {
  // NOTE (Yuang Liu): The peer endpoints will be obtained with the assumption
  // that mp part is always on inner side and pp part is always on outer side.
  // TODO(fleet exe dev): The peer endpoints could be configured by users.
  PADDLE_ENFORCE_EQ(
      config_.pp_degree * config_.mp_degree, config_.nranks,
      platform::errors::InvalidArgument(
          "The mp_degree multiplies pp_degree is not equal with nranks"));
  std::unique_ptr<framework::ProgramDesc> comm_init_program(
      new framework::ProgramDesc());
  framework::BlockDesc *comm_init_block = comm_init_program->MutableBlock(0);
  if (config_.mp_degree > 1) {
    PADDLE_ENFORCE_GE(
        config_.mp_ring_id, 0,
        platform::errors::InvalidArgument(
            "mp ring id must be provided for inference under mp."));
    VLOG(3) << "Init comm group for mp.";
    std::vector<std::string> peer_endpoints;
    for (int64_t
             idx = (config_.local_rank / config_.mp_degree) * config_.mp_degree,
             i = 0;
         i < config_.mp_degree; ++idx, ++i) {
      if (config_.trainer_endpoints[idx] == config_.current_endpoint) {
        continue;
      }
      peer_endpoints.emplace_back(config_.trainer_endpoints[idx]);
    }
    // get nranks in a mp group and inner group rank for local rank
    int64_t mp_group_nranks = config_.nranks / config_.pp_degree;
    int64_t mp_group_rank = config_.local_rank % config_.mp_degree;
    InsertCommOp("mp_comm_id", mp_group_nranks, mp_group_rank, peer_endpoints,
                 comm_init_block, config_.mp_ring_id);
  }
  if (config_.pp_degree) {
    // NOTE: the last pp stage doesn't need init pp comm
    VLOG(3) << "Init comm group for pp.";
    if (config_.local_rank - config_.mp_degree >= 0) {
      PADDLE_ENFORCE_EQ(config_.pp_upstream_ring_id >= 0, true,
                        platform::errors::InvalidArgument(
                            "pp upstream ring id must be provided for "
                            "non-first pp stage if inference under pp."));
      // not the first pp stage, has upstream
      std::vector<std::string> upstream_peer_endpoints;
      upstream_peer_endpoints.emplace_back(
          config_.trainer_endpoints[config_.local_rank - config_.mp_degree]);
      InsertCommOp("pp_upstream_comm_id", 2, 1, upstream_peer_endpoints,
                   comm_init_block, config_.pp_upstream_ring_id);
    }

    if (config_.local_rank + config_.mp_degree < config_.nranks) {
      PADDLE_ENFORCE_EQ(config_.pp_downstream_ring_id >= 0, true,
                        platform::errors::InvalidArgument(
                            "pp downstream ring id must be provided for "
                            "non-last pp stage if inference under pp."));
      // not the last pp stage, has downstream
      std::vector<std::string> downstream_peer_endpoints;
      downstream_peer_endpoints.emplace_back(
          config_.trainer_endpoints[config_.local_rank + config_.mp_degree]);
      InsertCommOp("pp_downstream_comm_id", 2, 0, downstream_peer_endpoints,
                   comm_init_block, config_.pp_downstream_ring_id);
    }
  }
  framework::NaiveExecutor e(place_);
  e.CreateVariables(*comm_init_program, 0, true, scope_.get());
  e.Prepare(scope_.get(), *comm_init_program, 0, false);
  e.Run();
  VLOG(3) << "Comm init successful.";
  return true;
}

void DistModel::InsertCommOp(std::string tmp_var_name, int nranks, int rank,
                             const std::vector<std::string> &peer_endpoints,
                             framework::BlockDesc *block, int ring_id) {
  /*
   * tmp_var_name: the var name for var comm_id
   * nranks: number of total ranks
   * rank: the rank of local rank in the comm group
   * peer_endpoints: peer's endpoints
   * block: the block where to insert the comm ops
   * ring_id: the ring_id to be inited
   */
  std::string &endpoint = config_.current_endpoint;
  std::stringstream ss;
  ss << "Init comm with tmp var: " << tmp_var_name
     << ". The ring id is: " << ring_id << ". The group has: " << nranks
     << " ranks. Current rank in the group is: " << rank
     << ". The endpoint is: " << endpoint << ". Peer endpoints are: ";
  for (auto ep : peer_endpoints) {
    ss << ep << ", ";
  }
  VLOG(3) << ss.str();
  if (config_.place == "GPU") {
    framework::VarDesc *new_var = block->Var(tmp_var_name);
    new_var->SetType(framework::proto::VarType::RAW);
    new_var->SetPersistable(true);
    framework::OpDesc *gen_nccl_id_op = block->AppendOp();
    gen_nccl_id_op->SetType("c_gen_nccl_id");
    gen_nccl_id_op->SetOutput("Out", {tmp_var_name});
    gen_nccl_id_op->SetAttr("rank", rank);
    gen_nccl_id_op->SetAttr("endpoint", config_.current_endpoint);
    gen_nccl_id_op->SetAttr("other_endpoints", peer_endpoints);
    gen_nccl_id_op->SetAttr("ring_id", ring_id);
    gen_nccl_id_op->SetAttr("op_role",
                            static_cast<int>(framework::OpRole::kForward));
    gen_nccl_id_op->CheckAttrs();
    framework::OpDesc *comm_init_op = block->AppendOp();
    comm_init_op->SetType("c_comm_init");
    comm_init_op->SetInput("X", {tmp_var_name});
    comm_init_op->SetAttr("rank", rank);
    comm_init_op->SetAttr("nranks", nranks);
    comm_init_op->SetAttr("ring_id", ring_id);
    comm_init_op->SetAttr("op_role",
                          static_cast<int>(framework::OpRole::kForward));
    comm_init_op->CheckAttrs();
  } else {
    LOG(WARNING) << "DistModelInf doesn't init comm.";
    // TODO(fleet exe dev): comm init for more devices
  }
}

bool DistModel::PrepareScope() {
  scope_.reset(new framework::Scope());
  return true;
}

bool DistModel::PrepareProgram() {
  if (!LoadProgram()) {
    return false;
  }
  if (!LoadParameters()) {
    return false;
  }
  return true;
}

bool DistModel::LoadProgram() {
  VLOG(3) << "Loading program from " << config_.model_dir;
  PADDLE_ENFORCE_NE(config_.model_dir, "", platform::errors::InvalidArgument(
                                               "Model dir must be provided."));
  std::string model_path = config_.model_dir + ".pdmodel";
  framework::proto::ProgramDesc program_proto;
  std::string pb_content;
  // Read binary
  std::ifstream fin(model_path, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fin.is_open()), true,
      platform::errors::NotFound(
          "Cannot open file %s, please confirm whether the file is normal.",
          model_path));
  fin.seekg(0, std::ios::end);
  pb_content.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(pb_content.at(0)), pb_content.size());
  fin.close();
  program_proto.ParseFromString(pb_content);
  VLOG(5) << pb_content;
  program_.reset(new framework::ProgramDesc(program_proto));
  return true;
}

bool DistModel::LoadParameters() {
  VLOG(3) << "Loading parameters from " << config_.model_dir;
  PADDLE_ENFORCE_NOT_NULL(program_.get(),
                          platform::errors::PreconditionNotMet(
                              "The program should be loaded first."));
  const auto &global_block = program_->MutableBlock(0);

  // create a temporary program to load parameters.

  std::unique_ptr<framework::ProgramDesc> load_program(
      new framework::ProgramDesc());
  framework::BlockDesc *load_block = load_program->MutableBlock(0);
  std::vector<std::string> params;

  for (auto *var : global_block->AllVars()) {
    if (IsPersistable(var)) {
      VLOG(3) << "persistable variable's name: " << var->Name();
      framework::VarDesc *new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());
      new_var->SetLoDLevel(var->GetLoDLevel());
      new_var->SetPersistable(true);
      params.push_back(new_var->Name());
      // NOTE: if the params are stored in different files, 'load' op should be
      // added here
    }
  }

  std::string param_path = config_.model_dir + ".pdiparams";
  // sort paramlist to have consistent ordering
  std::sort(params.begin(), params.end());
  // append just the load_combine op
  framework::OpDesc *op = load_block->AppendOp();
  op->SetType("load_combine");
  op->SetOutput("Out", params);
  op->SetAttr("file_path", {param_path});
  op->CheckAttrs();

  framework::NaiveExecutor e(place_);
  // Create all persistable variables in root scope to load them from ckpt.
  // Other non-persistable variables will be created in the micro scope
  // managed by fleet executor.
  e.CreateVariables(*program_, 0, true, scope_.get());
  e.Prepare(scope_.get(), *load_program, 0, false);
  e.Run();
  VLOG(3) << "After loading there are " << scope_->LocalVarNames().size()
          << " vars.";

  return true;
}

bool DistModel::PrepareFleetExe() {
  task_node_.reset(new TaskNode(program_.get(), config_.local_rank));
  if (config_.local_rank - config_.mp_degree >= 0) {
    task_node_->AddUpstreamTask(config_.local_rank - config_.mp_degree);
  }
  if (config_.local_rank + config_.mp_degree < config_.nranks) {
    task_node_->AddDownstreamTask(config_.local_rank + config_.mp_degree);
  }
  task_node_->SetType("Compute");
  task_node_->Init();
  executor_desc_ = FleetExecutorDesc();
  executor_desc_.set_cur_rank(config_.local_rank);
  std::unordered_map<int64_t, int64_t> id_to_rank;
  for (int i = 0; i < config_.nranks; ++i) {
    RankInfo *rank_info = executor_desc_.add_cluster_info();
    rank_info->set_rank(i);
    rank_info->set_ip_port(config_.trainer_endpoints[i]);
    id_to_rank.insert({i, i});
  }
  fleet_exe.reset(new FleetExecutor(executor_desc_));
  fleet_exe->Init("inference", *(program_.get()), scope_.get(), place_, 1,
                  {task_node_.get()}, id_to_rank);
  return true;
}

bool DistModel::PrepareFeedAndFetch() {
  for (auto *op : program_->Block(0).AllOps()) {
    if (op->Type() == "feed") {
      VLOG(3) << "feed op with feed var: " << op->Output("Out")[0];
      int idx = BOOST_GET_CONST(int, op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
      idx_to_feeds_[idx] = op->Output("Out")[0];
    } else if (op->Type() == "fetch") {
      VLOG(3) << "fetch op with fetch var: " << op->Input("X")[0];
      int idx = BOOST_GET_CONST(int, op->GetAttr("col"));
      if (fetches_.size() <= static_cast<size_t>(idx)) {
        fetches_.resize(idx + 1);
      }
      fetches_[idx] = op;
      id_to_fetches_[idx] = op->Input("X")[0];
    }
  }
  return true;
}

void DistModel::Run(const std::vector<DistModelTensor> &input_data,
                    std::vector<DistModelTensor> *output_data) {
  /* TODO(fleet exe dev): implement this funct */
}

}  // namespace distributed
}  // namespace paddle
