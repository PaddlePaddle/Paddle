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
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"
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

 private:
  std::vector<std::unique_ptr<OperatorBase>> ops_;
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

void LinearListView::Initialize(const ProgramDesc* pdesc) {
  // get a LinearView of ProgramDesc
  for (auto& block_desc : pdesc->blocks()) {
    for (auto& op_desc : block_desc.ops()) {
      ops_.emplace_back(OpRegistry::CreateOp(op_desc));
    }
  }
}

void GraphView::Initialize(const ProgramDesc* pdesc) {
  // get a GraphView of ProgramDesc
}

struct Device {
  platform::CPUDeviceContext* cpu_device_context;
#ifndef PADDLE_ONLY_CPU
  platform::CUDADeviceContext* cuda_device_context;
#endif

#ifndef PADDLE_ONLY_CPU
  Device(platform::CPUDeviceContext* cpu, platform::CUDADeviceContext* gpu)
      : cpu_device_context(cpu), cuda_device_context(gpu) {}
#else
  explicit Device(platform::CPUDeviceContext* cpu) : cpu_device_context(cpu) {}
#endif
};

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl(Scope* scope, const Device* device, const ProgramDesc* pdesc,
               bool is_linear)
      : scope_(scope),
        device_(device),
        program_desc_(pdesc),
        view_(ProgramDescView::Create(is_linear)) {}

  virtual ~ExecutorImpl() {
    if (view_) delete view_;
  }

  void Run() override;

  void Initialize();

 private:
  Scope* scope_;
  const Device* device_;
  const ProgramDesc* program_desc_;
  ProgramDescView* view_;
};

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

platform::CPUDeviceContext* GetCPUDeviceContext(
    const platform::CPUPlace& place) {
  static std::unique_ptr<platform::CPUDeviceContext> g_cpu_device_context =
      make_unique<platform::CPUDeviceContext>(place);
  return g_cpu_device_context.get();
}

#ifndef PADDLE_ONLY_CPU
platform::CUDADeviceContext* GetCUDADeviceContext(
    const platform::GPUPlace& place) {
  static std::unique_ptr<platform::CUDADeviceContext> g_cuda_device_context =
      make_unique<platform::CUDADeviceContext>(place);
  return g_cuda_device_context.get();
}
#endif

Device* GetDevice(const platform::Place& place) {
  platform::CPUPlace cpu_place;
#ifndef PADDLE_ONLY_CPU
  if (platform::is_gpu_place(place)) {
    platform::GPUPlace gpu_place = boost::get<platform::GPUPlace>(place);
    static std::unique_ptr<Device> g_device = make_unique<Device>(
        GetCPUDeviceContext(cpu_place), GetCUDADeviceContext(gpu_place));
    return g_device.get();
  } else {
    static std::unique_ptr<Device> g_device =
        make_unique<Device>(GetCPUDeviceContext(cpu_place), nullptr);
    return g_device.get();
  }
#else
  static std::unique_ptr<Device> g_device =
      make_unique<Device>(GetCPUDeviceContext(cpu_place));
  return g_device.get();
#endif
}

framework::Scope* GetScope() {
  static std::unique_ptr<framework::Scope> g_scope =
      make_unique<framework::Scope>();
  return g_scope.get();
}

Executor* NewLocalExecutor(const platform::Place& place,
                           const ProgramDesc& pdesc, bool is_linear) {
  return new ExecutorImpl(GetScope(), GetDevice(place), &pdesc, is_linear);
}

void ExecutorImpl::Run() {
  // operators running
  scope_->NewVar();
  device_->cpu_device_context->Wait();
#ifndef PADDLE_ONLY_CPU
  if (device_->cuda_device_context) {
    device_->cuda_device_context->Wait();
  }
#endif
}

void ExecutorImpl::Initialize() {
  // Initialize the ProgramDescView
  view_->Initialize(program_desc_);
}

}  // namespace framework
}  // namespace paddle
