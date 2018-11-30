/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/framework/collective_executor.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {

CollectiveExecutor::CollectiveExecutor(
    Scope& scope,
    const platform::Place& place)
    : root_scope_(scope), place_(place) {}

void CollectiveExecutor::SetNCCLId(const NCCLInfo& nccl_info) {
  // set nccl id from python
  LOG(ERROR) << "set nccl id";
  nccl_info_.nccl_id_ = nccl_info.nccl_id_;
}

NCCLInfo CollectiveExecutor::GetNCCLId() {
  PADDLE_ENFORCE(platform::dynload::ncclGetUniqueId(
      &(nccl_info_.nccl_id_)));
  return nccl_info_;
}

void CollectiveExecutor::InitNCCL() {
  // initialize comm_t and stream for nccl
  // init nccl
  platform::dynload::ncclCommInitRank(
      &(nccl_info_.comm_),
      nccl_info_.global_ranks_,
      nccl_info_.nccl_id_,
      nccl_info_.my_global_rank_);
  return;
}

void CollectiveExecutor::SetRankInfo(const int local_rank,
                                     const int global_rank,
                                     const int ranks) {
  // set local rank and global rank from python
  nccl_info_.local_rank_ = local_rank;
  nccl_info_.my_global_rank_ = global_rank;
  nccl_info_.global_ranks_ = ranks;
  LOG(ERROR) << "local rank: " << local_rank << " global rank: "
             << global_rank << " total rank " << ranks;
  PADDLE_ENFORCE(cudaSetDevice(local_rank));
  PADDLE_ENFORCE(cudaStreamCreate(&(nccl_info_.stream_)));
  return;
}

void CollectiveExecutor::SynchronizeModel(
    const ProgramDesc& program, int root_rank, Scope* scope) {
  // should check root_rank here
  LOG(ERROR) << "begin to run synchronize model";
  auto& block = program.Block(0);
  LOG(ERROR) << "block 0 found";
  for (auto& var : block.AllVars()) {
    LOG(ERROR) << "begin to process variable desc: " << var->Name();
    if (var->Persistable()) {
      LOG(ERROR) << "hehe";
      auto vari = scope->FindVar(var->Name());
      if (var == nullptr) {
        LOG(ERROR) << "nullptr of " << var->Name();
      }
      LoDTensor * tensor = vari->GetMutable<LoDTensor>();
      if (!tensor->IsInitialized()) {
        LOG(ERROR) << "Not inited!!!";
      }
      int32_t total_size = tensor->numel();
      LOG(ERROR) << "tensor name: " << var->Name() <<
          " tensor size: " << total_size;
      platform::dynload::ncclBcast(
          reinterpret_cast<void *>(tensor->data<float>()),
          total_size,
          ncclFloat,
          root_rank,
          nccl_info_.comm_,
          nccl_info_.stream_);
      LOG(ERROR) << "do bcast done.";
      cudaStreamSynchronize(nccl_info_.stream_);
      LOG(ERROR) << "sync done.";
    }
  }
  return;
}

// run and do all reduce here, that's it
// we can wrap different synchronous gradient sync algo into collective sync
// collective sync position should be assigned from python
std::vector<float> CollectiveExecutor::RunFromFile(
    const ProgramDesc& main_program,
    const std::string& data_feed_desc_str,
    const std::vector<std::string> & filelist,
    const std::vector<std::string> & fetch_names) {
  /*
  DataFeedDesc data_feed_desc;
  google::protobuf::TextFormat::ParseFromString(
      data_feed_desc_str, &data_feed_desc);
  std::shared_ptr<DataFeed> reader =
      DataFeedFactory::CreateDataFeed(data_feed_desc.name());
  reader->Init(data_feed_desc);
  reader->SetFileList(filelist);
  */
  // set worker here

  // run ff ops
  // run bp ops

  // run nccl all reduce here, maybe async from some point


  // run opt ops
  // until we don't have anymore files
  std::vector<float> res;
  return res;
}

}  // namespace framework
}  // namespace paddle

