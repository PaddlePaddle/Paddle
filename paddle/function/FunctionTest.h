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

#include "Function.h"
#include "paddle/math/Vector.h"
#include "paddle/math/tests/TensorCheck.h"

namespace paddle {

class FunctionCompare {
public:
  FunctionCompare(const std::string& name, const FuncConfig& config)
      : cpu(FunctionBase::funcRegistrar_.createByType(name + "-CPU")),
        gpu(FunctionBase::funcRegistrar_.createByType(name + "-GPU")) {
    cpu->init(config);
    gpu->init(config);
  }

  void cmpWithArg(const BufferArgs& inputs,
                  const BufferArgs& outputs,
                  const BufferArgs& inouts) {
    // init cpu and gpu arguments
    auto initArgs = [=](
        BufferArgs& cpuArgs, BufferArgs& gpuArgs, const BufferArgs& inArgs) {
      /// leave it empty to pass the compile of ContextProjectionTest
      /// Daoyuan is working on FunctionTest
      /// and I will further merge with it
    };
    initArgs(cpuInputs, gpuInputs, inputs);
    initArgs(cpuOutputs, gpuOutputs, outputs);

    // function calculate
    cpu->calc(cpuInputs, cpuOutputs);
    gpu->calc(gpuInputs, gpuOutputs);

    // check outputs and inouts
    auto checkArgs = [=](const BufferArgs& cpuArgs, const BufferArgs& gpuArgs) {
      /// leave it open
    };
    checkArgs(cpuOutputs, gpuOutputs);
  }

  std::shared_ptr<FunctionBase> getCpuFunction() const { return cpu; }

  std::shared_ptr<FunctionBase> getGpuFunction() const { return gpu; }

protected:
  std::shared_ptr<FunctionBase> cpu;
  std::shared_ptr<FunctionBase> gpu;
  std::vector<CpuMemHandlePtr> cpuMemory;
  std::vector<GpuMemHandlePtr> gpuMemory;
  BufferArgs cpuInputs;
  BufferArgs cpuOutputs;
  BufferArgs cpuInouts;
  BufferArgs gpuInputs;
  BufferArgs gpuOutputs;
  BufferArgs gpuInouts;
};

}  // namespace paddle
