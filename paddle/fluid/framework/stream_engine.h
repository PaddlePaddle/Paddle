// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <memory>
#include "operator.h"
#include "paddle/fluid/framework/ir/parallel_schedule_pass.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/cuda_api.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

using platform::CudaAPI;
using framework::ir::ParallelMeta;

struct StreamParallelStuff {
  CudaAPI::stream_t stream;
  CudaAPI::event_t event;
};
/*
 * An wrapper of operator, to enable setting stream and events to sync
 * externally.
 */
class StreamOperation {
 public:
  StreamOperation(std::unique_ptr<OperatorBase>&& op, Scope* scope,
                  platform::Place place)
      : op_(std::move(op)), scope_(scope), place_(place) {}

  OperatorBase* op() { return op_.get(); }
  // Set the stream the operator runs on.
  void SetStream(platform::CudaAPI::stream_t stream) { stream_ = stream; }

  // Set the events need to sync.
  void SetInputEvents(const std::vector<CudaAPI::event_t>& events) {
    input_events_ = events;
  }

  void SetOutputEvents(const std::vector<CudaAPI::event_t>& events) {
    output_events_ = events;
  }

  // Sync the inputs, make sure the inputs are valid.
  void SyncInputs() {
    for (auto event : input_events_) {
      LOG(INFO) << "sync input event " << event;
      CudaAPI::SyncEvent(event);
    }
  }

  void RecordOutputs() {
    for (auto event : output_events_) {
      LOG(INFO) << "record output event " << event;
      CudaAPI::RecordEvent(event, stream_);
    }
  }

  void Run() { op_->Run(*scope_, place_); }

  void RunAsync() {
    SyncInputs();

    if (dynamic_cast<OperatorWithKernel*>(op_.get())) {
      RunKernel();
    } else {
      // Run OperatorBase, that will only need the CPU place.
      op_->Run(*scope_, place_);
    }

    RecordOutputs();
  }

  void RunKernel() {
    cudaDeviceSynchronize();
    PADDLE_ENFORCE(stream_);
    auto* kernel_p = static_cast<OperatorWithKernel*>(op_.get());
    if (!runtime_context_) {
      runtime_context_.reset(
          new RuntimeContext(op_->Inputs(), op_->Outputs(), *scope_));
      cuda_device_context_.reset(new platform::CUDADeviceContext(
          boost::get<platform::CUDAPlace>(place_), stream_));
      const auto& kernel_iter =
          OperatorWithKernel::AllOpKernels().find(op_->Type());
      PADDLE_ENFORCE(kernel_iter != OperatorWithKernel::AllOpKernels().end());
      const auto& kernel_map = kernel_iter->second;

      execution_context_.reset(
          new ExecutionContext(*kernel_p, *scope_, *cuda_device_context_,
                               *runtime_context_, nullptr));
      kernel_type_.reset(new OpKernelType(
          static_cast<OperatorWithKernel*>(op_.get())->GetExpectedKernelType(
              *execution_context_)));
      auto kernel = kernel_map.find(*kernel_type_);

      PADDLE_ENFORCE(kernel != kernel_map.end(), "no kernel found for %s",
                     *kernel_type_);
      kernel_ = kernel->second;

      // auto kernel_configs = kernel_p->GetKernelConfig(kernel_type_);
      // Infer shape.
      infer_shape_context_.reset(new RuntimeInferShapeContext(
          *kernel_p, *exec_scope_, *runtime_context_));
    }

    std::vector<std::string> transfered_inplace_vars;
    TransferScope(&transfered_inplace_vars);

    kernel_p->InferShape(infer_shape_context_.get());

    // execute the kernel
    kernel_(*execution_context_);

    if (!transfered_inplace_vars.empty()) {
      kernel_p->TransferInplaceVarsBack(*scope_, transfered_inplace_vars,
                                        *transfer_scope_);
    }

    cudaDeviceSynchronize();
  }

  std::string type() const { return op_->Type(); }

  // Copy data across device automatically.
  // TODO(Superjomn) Improve the performance here.
  void TransferScope(std::vector<std::string>* transfered_inplace_vars) {
    auto* kernel_p = static_cast<OperatorWithKernel*>(op_.get());
    transfer_scope_ =
        kernel_p->PrepareData(*scope_, *kernel_type_, transfered_inplace_vars,
                              runtime_context_.get());
    exec_scope_ = transfer_scope_ ? transfer_scope_ : scope_;
  }

 private:
  // stream related.
  cudaStream_t stream_{0};
  std::vector<CudaAPI::event_t> input_events_;
  std::vector<CudaAPI::event_t> output_events_;

  // execution info.
  std::unique_ptr<OperatorBase> op_;
  framework::Scope* scope_{nullptr};
  framework::Scope* exec_scope_{nullptr};
  framework::Scope* transfer_scope_{nullptr};
  platform::Place place_;
  std::unique_ptr<OpKernelType> kernel_type_;
  OperatorWithKernel::OpKernelFunc kernel_;
  // contexts.
  std::unique_ptr<RuntimeContext> runtime_context_;
  std::unique_ptr<ExecutionContext> execution_context_;
  std::unique_ptr<platform::CUDADeviceContext> cuda_device_context_;
  std::unique_ptr<RuntimeInferShapeContext> infer_shape_context_;
};

/*
 * An operator execution engine with GPU streaming parallel support.
 * It takes a list of operators as input, and run them wit multiple stream.
 */
class StreamEngine final {
 public:
  StreamEngine(std::vector<std::unique_ptr<OperatorBase>>* ops, Scope* scope,
               platform::Place place, const ParallelMeta& parallel_meta) {
    // Get number of streams.
    std::set<int> stream_set;

    for (int id : parallel_meta.StreamIds()) {
      parallel_stuff_.emplace(id,
                              StreamParallelStuff{CudaAPI::CreateStream(),
                                                  CudaAPI::CreateEvent(true)});
    }

    for (auto& op : *ops) {
      // LOG(INFO) << "creating stream operation " << op->Type();
      operations_.emplace_back(
          new StreamOperation(std::move(op), scope, place));
      auto& operation = operations_.back();
      // Prepare input events
      std::vector<CudaAPI::event_t> input_events, output_events;

      const auto op_key =
          GenOpKey(operation->op()->Type(), operation->op()->InputVars(),
                   operation->op()->OutputVars(true));
      auto op_stream_id = parallel_meta.GetStreamId(op_key);
      operation->SetStream(parallel_stuff_.at(op_stream_id).stream);

      for (int id : parallel_meta.GetInputDependEventIds(op_key)) {
        input_events.push_back(parallel_stuff_.at(id).event);
      }
      for (int id : parallel_meta.GetOutputDependEventIds(op_key)) {
        output_events.push_back(parallel_stuff_.at(id).event);
      }
      operations_.back()->SetInputEvents(input_events);
      operations_.back()->SetOutputEvents(output_events);
    }
  }

  void Run(bool async = true) {
    for (auto& op : operations_) {
      LOG(INFO) << "running operation " << op->type();
      if (async) {
        op->RunAsync();
      } else {
        op->Run();
      }
    }
  }

 private:
  std::unordered_map<int, StreamParallelStuff> parallel_stuff_;
  std::vector<std::unique_ptr<StreamOperation>> operations_;
};

}  // namespace framework
}  // namespace paddle

#endif
