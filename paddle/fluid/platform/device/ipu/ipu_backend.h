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

#include <popart/devicemanager.hpp>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/ipu/ipu_compiler.h"
#include "paddle/fluid/platform/device/ipu/ipu_device.h"
#include "paddle/fluid/platform/device/ipu/ipu_executor.h"
#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"
#include "paddle/fluid/platform/device/ipu/ipu_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace platform {
namespace ipu {

// IpuBackend is the center of paddle-ipu, its function include:
//   1. Compile paddle model to popart model
//   2. Run popart model, inference or training
//   3. Request and release device
//   4. Other helper function
class IpuBackend {
 public:
  static IpuBackend *GetInstance();

 public:
  IpuBackend();
  ~IpuBackend();

  // what compile does include(call compiler_):
  //   1. map paddle-op -> poart op
  //   2. construct popart onnx compute graph
  void Compile(Graph *graph, const std::vector<std::string> &feed_list,
               const std::vector<std::string> &fetch_list);

  // what run does include:
  //   1. construct forward onnx graph
  //   2. graph-level optimization
  //   3. autodiff
  void Run(const std::vector<const Tensor *> &inputs,
           const std::vector<Tensor *> &outputs,
           const framework::ExecutionContext &ctx);

  // detach IPU manually
  void Detach();

  // reset manually
  // call it before destruct works
  void Reset();

  void SetScope(const Scope &scope);
  const Scope *GetScope() { return scope_; }
  void SetIpuStrategy(const IpuStrategy &strategy);
  const IpuStrategy *GetIpuStrategy() { return ipu_strategy_; }
  void SetCustomOps(const std::vector<IpuCustomOpIdentifier> &custom_ops);

  // save compiled model to onnx
  void SaveMoldeProto(const std::string &path);

 private:
  void Prepare();

 private:
  std::unique_ptr<Compiler> compiler_;
  std::unique_ptr<Executor> executor_;
  bool is_compiled_ = false;
  bool is_prepared_ = false;

  // not own
  const Scope *scope_ = nullptr;
  const IpuStrategy *ipu_strategy_ = nullptr;

 private:
  // time record for IpuBackend::Run
  std::unique_ptr<platform::Timer> timer_;

  DISABLE_COPY_AND_ASSIGN(IpuBackend);
};

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
