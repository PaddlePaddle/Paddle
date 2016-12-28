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

  void cmpWithArg(const Arguments& inputs,
                  const Arguments& outputs,
                  const Arguments& inouts) {
    // init cpu and gpu arguments
    auto initArgs = [=](
        Arguments& cpuArgs, Arguments& gpuArgs, const Arguments& inArgs) {
      for (auto arg : inArgs) {
        size_t size = sizeof(real);
        for (auto dim : arg.dims_) {
          size *= dim;
        }
        cpuMemory.emplace_back(std::make_shared<CpuMemoryHandle>(size));
        gpuMemory.emplace_back(std::make_shared<GpuMemoryHandle>(size));
        cpuArgs.emplace_back(
            Tensor((real*)cpuMemory.back()->getBuf(), arg.dims_));
        gpuArgs.emplace_back(
            Tensor((real*)gpuMemory.back()->getBuf(), arg.dims_));

        // will use an api to refactor this code.
        CpuVector cpuVector(size / sizeof(real),
                            (real*)cpuArgs.back().getData());
        GpuVector gpuVector(size / sizeof(real),
                            (real*)gpuArgs.back().getData());
        cpuVector.uniform(0.001, 1);
        gpuVector.copyFrom(cpuVector);
      }
    };
    initArgs(cpuInputs, gpuInputs, inputs);
    initArgs(cpuOutputs, gpuOutputs, outputs);
    initArgs(cpuInouts, gpuInouts, inouts);

    // function calculate
    cpu->calc(cpuInputs, cpuOutputs, cpuInouts);
    gpu->calc(gpuInputs, gpuOutputs, gpuInouts);

    // check outputs and inouts
    auto checkArgs = [=](const Arguments& cpuArgs, const Arguments& gpuArgs) {
      for (size_t i = 0; i < cpuArgs.size(); i++) {
        auto cpu = cpuArgs[i];
        auto gpu = gpuArgs[i];
        size_t size = 1;
        for (auto dim : cpu.dims_) {
          size *= dim;
        }
        CpuVector cpuVector(size, (real*)cpu.getData());
        GpuVector gpuVector(size, (real*)gpu.getData());

        autotest::TensorCheckErr(cpuVector, gpuVector);
      }
    };
    checkArgs(cpuOutputs, gpuOutputs);
    checkArgs(cpuInouts, gpuInouts);
  }

protected:
  std::shared_ptr<FunctionBase> cpu;
  std::shared_ptr<FunctionBase> gpu;
  std::vector<CpuMemHandlePtr> cpuMemory;
  std::vector<GpuMemHandlePtr> gpuMemory;
  Arguments cpuInputs;
  Arguments cpuOutputs;
  Arguments cpuInouts;
  Arguments gpuInputs;
  Arguments gpuOutputs;
  Arguments gpuInouts;
};

}  // namespace paddle
