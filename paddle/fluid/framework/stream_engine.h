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
#include "cuda_profiler_api.h"
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

struct StreamRecorder {
  StreamRecorder(CudaAPI::event_t start, CudaAPI::event_t end,
                 CudaAPI::stream_t stream)
      : start(start), end(end), stream(stream) {
    CudaAPI::RecordEvent(start, stream);
  }

  void Touch() {}

  ~StreamRecorder() { CudaAPI::RecordEvent(end, stream); }

  CudaAPI::event_t start;
  CudaAPI::event_t end;
  CudaAPI::stream_t stream;
};

/*
 * An wrapper of operator, to enable setting stream and events to sync
 * externally.
 */
class StreamOperation {
 public:
  struct ProfileRecord {
    ProfileRecord() {
      cudaEventCreate(&start_event);
      cudaEventCreate(&end_event);
    }

    void UpdateDuration() {
      PADDLE_ENFORCE(cudaEventElapsedTime(&duration, start_event, end_event));
    }

    float start_time{0.};
    float duration;
    CudaAPI::event_t start_event;
    CudaAPI::event_t end_event;
  };

  StreamOperation(std::unique_ptr<OperatorBase>&& op, Scope* scope,
                  platform::Place place, bool enable_profiler = true)
      : op_(std::move(op)),
        scope_(scope),
        place_(place),
        enable_profiler_(enable_profiler) {}

  OperatorBase* op() { return op_.get(); }
  // Set the stream the operator runs on.
  void SetStream(platform::CudaAPI::stream_t stream) { stream_ = stream; LOG(INFO) << "op get stream " << stream; }

  // Set the events need to sync.
  void SetInputEvents(const std::vector<CudaAPI::event_t>& events) {
    input_events_ = events;
  }

  void SetOutputEvents(const std::vector<CudaAPI::event_t>& events) {
    output_events_ = events;
  }

  void Run() {
    //LOG(INFO) << "running normal op " << op_->Type();
    op_->SetIsCalledByExecutor(false);
    op_->Run(*scope_, place_);
  }

  void RunAsync();

  void GetProfilerInfo() { profile_record_.UpdateDuration(); }

 private:
  void GetKernel();
  void RunKernel();

  // Sync the inputs, make sure the inputs are valid.
  void SyncInputs();

  void RecordOutputs();

  std::string type() const { return op_->Type(); }

  // Copy data across device automatically.
  // TODO(Superjomn) Improve the performance here.
  void TransferScope(std::vector<std::string>* transfered_inplace_vars);

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
  // Just for get kernel type.
  std::unique_ptr<ExecutionContext> execution_context_;
  // For kernel execution.
  std::unique_ptr<ExecutionContext> runtime_execution_context_;
  std::unique_ptr<platform::CUDADeviceContext> cuda_device_context_;
  std::unique_ptr<RuntimeInferShapeContext> infer_shape_context_;
  // For profiler
  bool enable_profiler_{false};
  ProfileRecord profile_record_;
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
                   operation->op()->OutputVars(false));
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
    // cudaProfilerStart();
    for (auto& op : operations_) {
      // LOG(INFO) << "running operation " << op->type();
      if (async) {
        op->RunAsync();
      } else {
        op->Run();
      }
    }
    cudaDeviceSynchronize();
    // cudaProfilerStop();
  }

 private:
  std::unordered_map<int, StreamParallelStuff> parallel_stuff_;
  std::vector<std::unique_ptr<StreamOperation>> operations_;
};

}  // namespace framework
}  // namespace paddle

#endif
