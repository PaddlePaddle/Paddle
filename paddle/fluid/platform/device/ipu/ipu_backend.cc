/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/ipu/ipu_backend.h"
#include "paddle/fluid/platform/ipu/ipu_utils.h"

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace platform {
namespace ipu {

std::shared_ptr<IpuBackend> IpuBackend::instance_ = nullptr;

IpuBackend::IpuBackend() {
  compiler_ = std::make_shared<Compiler>();
  executor_ = std::make_unique<Executor>();
}

void IpuBackend::Clear() {
  executor_.reset();
  // detach device
  if (device_ != nullptr && device_->isAttached()) {
    device_->detach();
    device_.reset();
    device_ = nullptr;
  }
}

IpuBackend::~IpuBackend() { Clear(); }

std::shared_ptr<IpuBackend> IpuBackend::GetInstance() {
  if (!instance_) {
    instance_.reset(new IpuBackend());
  }
  return instance_;
}

// This api should only call from python, always return a new object
std::shared_ptr<IpuBackend> IpuBackend::GetNewInstance() {
  instance_.reset(new IpuBackend());
  return instance_;
}

void IpuBackend::Compile(framework::ir::Graph* graph,
                         const std::vector<std::string>& feed_list,
                         const std::vector<std::string>& fetch_list) {
  VLOG(10) << "enter IpuBackend::Compile";
  compiler_->InitInputs(graph, feed_list);
  compiler_->LowerWeights(graph, scope_);
  compiler_->LowerBody(graph);
  compiler_->InitOutputs(fetch_list);
  executor_->SetWeights(compiler_->GetWeights());
  VLOG(10) << "leave IpuBackend::Compile";
}

void IpuBackend::Run(const std::vector<const framework::Tensor*>& inputs,
                     const std::vector<framework::Tensor*>& outputs,
                     const framework::ExecutionContext& ctx) {
  Prepare();
  auto inputs_id = compiler_->GetInputs();
  auto outputs_id = compiler_->GetOutputs();
  executor_->Run(inputs_id, inputs, outputs_id, outputs, ctx);
}

void IpuBackend::Prepare() {
  if (is_prepared_) {
    return;
  } else {
    is_prepared_ = true;
  }
  // convert Model to fp16
  if (ipu_strategy_->enable_fp16) {
    compiler_->ConvertProtoToFp16();
  }
  auto proto = compiler_->GetModelProto();
  auto tensors = compiler_->GetTensors();
  auto outputs = compiler_->GetOutputs();
  executor_->Prepare(proto, tensors, outputs, device_);
}

void IpuBackend::SetScope(const framework::Scope& scope) {
  scope_ = &scope;
  executor_->SetScope(&scope);
}

void IpuBackend::SetIpuStrategy(const IpuStrategy& strategy) {
  ipu_strategy_ = &strategy;
  executor_->SetIpuStrategy(strategy);
  compiler_->SetIpuStrategy(strategy);
}

size_t IpuBackend::GetNumDevices() {
  // IpuModel
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (ipu_model) return 1;
  // Real dev
  size_t num_devices =
      popart::DeviceManager::createDeviceManager().enumerateDevices().size();
  PADDLE_ENFORCE_GT(
      num_devices, 0,
      platform::errors::Unavailable(
          "Do not found any IPU devices, please make "
          "sure Poplar sdk is enabled or enable ENV \"POPLAR_IPUMODEL=1\""));
  return num_devices;
}

std::vector<int> IpuBackend::GetDeviceIds() {
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (ipu_model) {
    return {0};
  }
  std::vector<int> device_ids;
  auto devices =
      popart::DeviceManager::createDeviceManager().enumerateDevices();
  PADDLE_ENFORCE_GT(
      devices.size(), 0,
      platform::errors::Unavailable("Do not found any IPU devices, please make "
                                    "sure Poplar sdk is enabled."));

  for (auto device : devices) {
    device_ids.push_back(device->getId());
  }

  return device_ids;
}

Device IpuBackend::GetDevice(int id) {
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (ipu_model) {
    std::map<std::string, std::string> deviceOpts{{"numIPUs", "1 "}};
    device_ = popart::DeviceManager::createDeviceManager().createIpuModelDevice(
        deviceOpts);
    Device device(*device_.get());
    return device;
  }
  size_t num_devices = GetNumDevices();
  if (id < 0 || id >= num_devices) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "device id %d is invalid, number devices is %d", id, num_devices));
  }
  std::shared_ptr<popart::DeviceInfo> popart_device_info =
      popart::DeviceManager::createDeviceManager().getDevice(
          popart::SyncPattern::Full, id);
  Device device(*popart_device_info.get());
  return device;
}

void IpuBackend::AttachDevice(int id) {
  // trick here
  // Compiler ipu is not same as the runtime ipu.
  VLOG(10) << "comile ipu id = " << id;
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (ipu_model) {
    return;
  }
  device_ = popart::DeviceManager::createDeviceManager().acquireAvailableDevice(
      UpperIpuNum());
  PADDLE_ENFORCE_NOT_NULL(
      device_, platform::errors::Unavailable("Can't attach IPU, ipu_num = %d.",
                                             UpperIpuNum()));
}

bool IpuBackend::DeviceIsAttached() { return device_ != nullptr; }

// num_ipus must be pow(2,n);
int IpuBackend::UpperIpuNum() {
  PADDLE_ENFORCE_GT(ipu_strategy_->num_ipus, 0,
                    platform::errors::Unavailable(
                        "The ipu num get is wrong, please make sure the "
                        "sharding or pipline parameter is right."));
  int i = 0;
  while (std::pow(2, i) < ipu_strategy_->num_ipus) {
    i++;
  }
  return std::pow(2, i);
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
