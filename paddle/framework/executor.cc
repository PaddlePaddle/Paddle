/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/executor.h"
#include <memory>
#include "paddle/framework/scope.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {

class LinearListView;
class GraphView;

// Immutable view of a ProgramDesc organized for efficient execution.
class ProgramDescView {
 public:
  virtual ~ProgramDescView() {}
  virtual void Initialize(const ProgramDesc*) = 0;
  static ProgramDescView* Create(bool is_linear);
};

class LinearListView : public ProgramDescView {
 public:
  void Initialize(const ProgramDesc*) override;
};

class GraphView : public ProgramDescView {
 public:
  void Initialize(const ProgramDesc*) override;
};

ProgramDescView* ProgramDescView::Create(bool is_linear) {
  if (is_linear) {
    return new LinearListView();
  } else {
    return new GraphView();
  }
}

void LinearListView::Initialize(const ProgramDesc*) {
  // get a LinearView of ProgramDesc
}

void GraphView::Initialize(const ProgramDesc*) {
  // get a GraphView of ProgramDesc
}

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl(Scope* scope, const platform::DeviceContext* ctx,
               const ProgramDesc* pdesc, bool is_linear)
      : scope_(scope),
        device_context_(ctx),
        program_desc_(pdesc),
        view_(ProgramDescView::Create(is_linear)) {}

  virtual ~ExecutorImpl() {
    if (view_) delete view_;
  }

  void Run() override;

  void Initialize();

 private:
  Scope* scope_;
  const platform::DeviceContext* device_context_;
  const ProgramDesc* program_desc_;
  ProgramDescView* view_;
};

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

platform::CPUDeviceContext* GetCPUDeviceContext(platform::CPUPlace& place) {
  static std::unique_ptr<platform::CPUDeviceContext> g_cpu_device_context =
      make_unique<platform::CPUDeviceContext>(place);
  return g_cpu_device_context.get();
}

#ifndef PADDLE_ONLY_CPU
platform::CUDADeviceContext* GetCUDADeviceContext(platform::GPUPlace& place) {
  static std::unique_ptr<platform::CUDADeviceContext> g_cuda_device_context =
      make_unique<platform::CUDADeviceContext>(place);
  return g_cuda_device_context.get();
}
#endif

framework::Scope* GetScope() {
  static std::unique_ptr<framework::Scope> g_scope =
      make_unique<framework::Scope>();
  return g_scope.get();
}

Executor* NewLocalExecutor(const platform::Place& place,
                           const ProgramDesc& pdesc, bool is_linear) {
  platform::DeviceContext* device_context = nullptr;
  if (platform::is_cpu_place(place)) {
    auto cpu_place = boost::get<platform::CPUPlace>(place);
    device_context = GetCPUDeviceContext(cpu_place);
  } else if (platform::is_gpu_place(place)) {
#ifndef PADDLE_ONLY_CPU
    auto gpu_place = boost::get<platform::GPUPlace>(place);
    device_context = GetCUDADeviceContext(gpu_place);
  }
#else
    PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
  }
#endif
  return new ExecutorImpl(GetScope(), device_context, &pdesc, is_linear);
}

void ExecutorImpl::Run() {
  // operators running
  scope_->NewVar();
  device_context_->Wait();
}

void ExecutorImpl::Initialize() {
  // Initialize the ProgramDescView
  view_->Initialize(program_desc_);
}

}  // namespace framework
}  // namespace paddle
