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

#include "paddle/fluid/framework/stream_engine.h"

namespace paddle {
namespace framework {

void StreamOperation::RunAsync() {
  op_->SetIsCalledByExecutor(false);
  SyncInputs();
  // cudaDeviceSynchronize();

  // if (!enable_profiler_) {
  //   StreamRecorder rcd(
  //       {profile_record_.start_event, profile_record_.end_event, stream_});
  //   rcd.Touch();

  //   if (dynamic_cast<OperatorWithKernel*>(op_.get())) {
  //     // LOG(INFO) << "running kernel " << op_->Type();
  //     RunKernel();
  //   } else {
  //     // LOG(INFO) << "running op " << op_->Type();
  //     // Run OperatorBase, that will only need the CPU place.
  //     op_->Run(*scope_, place_);
  //   }
  // } else {
  if (dynamic_cast<OperatorWithKernel*>(op_.get())) {
    // LOG(INFO) << "running kernel " << op_->Type();
    RunKernel();
  } else {
    // LOG(INFO) << "running op " << op_->Type();
    // Run OperatorBase, that will only need the CPU place.
    op_->Run(*scope_, place_);
  }
  // }

  RecordOutputs();
  // cudaStreamSynchronize(stream_);
}

void StreamOperation::RunKernel() {
  PADDLE_ENFORCE(platform::is_gpu_place(place_));
  // cudaDeviceSynchronize();
  PADDLE_ENFORCE(stream_);

  auto* kernel_p = static_cast<OperatorWithKernel*>(op_.get());
  if (!runtime_context_) {
    runtime_context_.reset(
        new RuntimeContext(op_->Inputs(), op_->Outputs(), *scope_));
    cuda_device_context_.reset(new platform::CUDADeviceContext(
        boost::get<platform::CUDAPlace>(place_), stream_));
    execution_context_.reset(new ExecutionContext(
        *kernel_p, *scope_, *cuda_device_context_, *runtime_context_, nullptr));

    GetKernel();

    auto kernel_configs = kernel_p->GetKernelConfig(*kernel_type_);

    runtime_execution_context_.reset(
        new ExecutionContext(*kernel_p, *scope_, *cuda_device_context_,
                             *runtime_context_, kernel_configs));
    // Infer shape.
    infer_shape_context_.reset(new RuntimeInferShapeContext(
        *kernel_p, *exec_scope_, *runtime_context_));
  }

  std::vector<std::string> transfered_inplace_vars;
  TransferScope(&transfered_inplace_vars);

  kernel_p->InferShape(infer_shape_context_.get());

  // execute the kernel
  kernel_(*runtime_execution_context_);

  if (!transfered_inplace_vars.empty()) {
    kernel_p->TransferInplaceVarsBack(*scope_, transfered_inplace_vars,
                                      *transfer_scope_);
  }
  // cudaDeviceSynchronize();
}

void StreamOperation::GetKernel() {
  const auto& kernel_iter =
      OperatorWithKernel::AllOpKernels().find(op_->Type());
  PADDLE_ENFORCE(kernel_iter != OperatorWithKernel::AllOpKernels().end());
  const auto& kernel_map = kernel_iter->second;

  kernel_type_.reset(new OpKernelType(
      static_cast<OperatorWithKernel*>(op_.get())->GetExpectedKernelType(
          *execution_context_)));
  auto kernel = kernel_map.find(*kernel_type_);

  PADDLE_ENFORCE(kernel != kernel_map.end(), "no kernel found for %s",
                 *kernel_type_);
  kernel_ = kernel->second;
}

void StreamOperation::TransferScope(
    std::vector<std::string>* transfered_inplace_vars) {
  auto* kernel_p = static_cast<OperatorWithKernel*>(op_.get());
  transfer_scope_ = kernel_p->PrepareData(
      *scope_, *kernel_type_, transfered_inplace_vars, runtime_context_.get());
  exec_scope_ = transfer_scope_ ? transfer_scope_ : scope_;
}

void StreamOperation::SyncInputs() {
  for (auto event : input_events_) {
    // LOG(INFO) << "sync input event " << event;
    // This will block the host thread.
    //CudaAPI::SyncEvent(event);
    cudaStreamWaitEvent(stream_, event, 0);
  }
}

void StreamOperation::RecordOutputs() {
  for (auto event : output_events_) {
    // LOG(INFO) << "record output event " << event;
    CudaAPI::RecordEvent(event, stream_);
    //cudaStreamWaitEvent(stream_, event, 0);
  }
}
}  // namespace framework
}  // namespace paddle
