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

#include "paddle/fluid/framework/multi_gpu_executor.h"

#include <thread>
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {

ExecutorWithAllReduce::ExecutorWithAllReduce(
    const platform::Place& place, std::unordered_set<std::string>* param_grads,
    NCCLContext* nccl_context)
    : Executor(place), param_grads_(param_grads) {
  int device_id = boost::get<platform::CUDAPlace>(place).device;
  comm_ = &nccl_context->comms_[device_id];
  io_ctx_ = nccl_context->ctxs_[device_id];
}

// TODO(yy): Move this function somewhere
ncclDataType_t ToNCCLDataType(std::type_index type) {
  if (type == typeid(float)) {  // NOLINT
    return ncclFloat;
  } else if (type == typeid(double)) {  // NOLINT
    return ncclDouble;
  } else if (type == typeid(int)) {  // NOLINT
    return ncclInt;
  } else {
    PADDLE_THROW("Not supported");
  }
}

void ExecutorWithAllReduce::RunOperators(const ExecutorPrepareContext* ctx,
                                         const Scope* local_scope) const {
  cudaSetDevice(boost::get<platform::CUDAPlace>(place_).device);

  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place_);
  cudaStream_t computation_stream =
      reinterpret_cast<const platform::CUDADeviceContext*>(dev_ctx)->stream();
  cudaStream_t all_reduce_stream = io_ctx_->stream();

  std::map<std::string, cudaEvent_t> computation_event;
  std::map<std::string, cudaEvent_t> all_reduce_event;
  for (auto& argu : *param_grads_) {
    PADDLE_ENFORCE(cudaEventCreateWithFlags(&computation_event[argu],
                                            cudaEventDisableTiming));
    PADDLE_ENFORCE(cudaEventCreateWithFlags(&all_reduce_event[argu],
                                            cudaEventDisableTiming));
  }

  for (auto& op : ctx->ops_) {
    for (auto& param2argu : op->Inputs()) {
      for (auto& argu : param2argu.second) {
        if (param_grads_->count(argu) != 0) {
          cudaStreamWaitEvent(computation_stream, all_reduce_event[argu], 0);
        }
      }
    }

    VLOG(4) << place_ << " " << op->DebugStringEx(local_scope);
    op->Run(*local_scope, place_);
    VLOG(3) << place_ << " " << op->DebugStringEx(local_scope);

    for (auto& param2argu : op->Outputs()) {
      for (auto& argu : param2argu.second) {
        if (param_grads_->count(argu) != 0) {
          LOG(INFO) << place_ << "Launch allreduce on " << argu;

          PADDLE_ENFORCE(
              cudaEventRecord(computation_event[argu], computation_stream));
          PADDLE_ENFORCE(cudaStreamWaitEvent(all_reduce_stream,
                                             computation_event[argu], 0));

          auto& tensor = local_scope->FindVar(argu)->Get<LoDTensor>();
          void* data = const_cast<void*>(tensor.data<void>());
          PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
              data, data, tensor.numel(), ToNCCLDataType(tensor.type()),
              ncclSum, *comm_, all_reduce_stream));

          PADDLE_ENFORCE(
              cudaEventRecord(all_reduce_event[argu], all_reduce_stream));
        }
      }
    }
  }

  cudaStreamSynchronize(computation_stream);
  cudaStreamSynchronize(all_reduce_stream);
  for (auto& argu : *param_grads_) {
    PADDLE_ENFORCE(cudaEventDestroy(computation_event[argu]));
    PADDLE_ENFORCE(cudaEventDestroy(all_reduce_event[argu]));
  }
}

MultiGPUExecutor::MultiGPUExecutor(
    const std::vector<platform::Place>& places,
    const std::unordered_set<std::string>& params)
    : nccl_ctx_(places) {
  for (auto& param : params) {
    param_grads_.insert(GradVarName(param));
  }
  for (auto& place : places) {
    exes_.push_back(
        framework::ExecutorWithAllReduce(place, &param_grads_, &nccl_ctx_));
    scopes_.push_back(new framework::Scope());
  }
}

void MultiGPUExecutor::Run(const ProgramDesc& prog, int block_id,
                           bool create_local_scope, bool create_vars) {
  // prepare prog in a single thread to avoid race
  auto* context = exes_[0].Prepare(prog, block_id);

  std::vector<std::thread> threads;
  threads.push_back(std::thread([&] {
    exes_[0].RunPreparedContext(context, scopes_[0], create_local_scope,
                                create_vars);
  }));
  threads.push_back(std::thread([&] {
    exes_[1].RunPreparedContext(context, scopes_[1], create_local_scope,
                                create_vars);
  }));

  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace framework
}  // namespace paddle
