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

#pragma once

#include <cmath>
#include <popart/devicemanager.hpp>
#include <popart/names.hpp>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/ipu/device.h"
#include "paddle/fluid/platform/ipu/ipu_compiler.h"
#include "paddle/fluid/platform/ipu/ipu_executor.h"
#include "paddle/fluid/platform/ipu/ipu_strategy.h"

namespace paddle {
namespace platform {
namespace ipu {

class IpuBackend {
  // IpuBackend is the center of paddle-ipu, its function include:
  //   1. Compile paddle model to popart model
  //   2. Run popart model, inference or training
  //   3. Request and release device
  //   4. Other helper function

 public:
  IpuBackend();
  ~IpuBackend();

  void Clear();

  // return if exsits, else create and return
  static std::shared_ptr<IpuBackend> GetInstance();

  // always return a new instance_
  static std::shared_ptr<IpuBackend> GetNewInstance();

  // what compile does include(call compiler_):
  //   1. map paddle-op -> poart op
  //   2. construct popart onnx compute graph
  void Compile(framework::ir::Graph *graph,
               const std::vector<std::string> &feed_list,
               const std::vector<std::string> &fetch_list);

  // what run does include:
  //   1. construct forward onnx graph
  //   2. graph-level optimization
  //   3. autodiff
  void Run(const std::vector<const framework::Tensor *> &inputs,
           const std::vector<framework::Tensor *> &outputs,
           const framework::ExecutionContext &ctx);

  Executor &GetExecutor() { return *executor_; }

  void SetScope(const framework::Scope &scope);
  const framework::Scope *GetScope() { return scope_; }
  void SetIpuStrategy(const IpuStrategy &strategy);
  const IpuStrategy *GetIpuStrategy() { return ipu_strategy_; }

  // Device
  size_t GetNumDevices();
  std::vector<int> GetDeviceIds();
  Device GetDevice(int id);
  void AttachDevice(int id);
  bool DeviceIsAttached();

 private:
  int UpperIpuNum();
  void Prepare();

 private:
  std::shared_ptr<Compiler> compiler_;
  std::unique_ptr<Executor> executor_;
  std::shared_ptr<popart::DeviceInfo> device_;
  bool is_prepared_ = false;

  // not own
  const framework::Scope *scope_ = nullptr;
  const IpuStrategy *ipu_strategy_ = nullptr;

 private:
  static std::shared_ptr<IpuBackend> instance_;
};

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
