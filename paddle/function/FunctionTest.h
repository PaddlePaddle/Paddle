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
#include "paddle/testing/TestUtil.h"

namespace paddle {

/**
 * \brief A class for comparing CPU and GPU implementations of Function.
 *
 *
 * Use case:
 *  // Initializes a test object, the corresponding cpu and gpu Function
 *  // are constructed according to FunctionName and FuncConfig.
 *  FunctionCompare test(FunctionName, FuncConfig);
 *  // Prepare inputs and outputs arguments.
 *  // Here the input and output can not contain real data,
 *  // only contains the argument type and shape.
 *  test.addInputs(input1);
 *  test.addInputs(input2);
 *  test.addOutputs(output1);
 *  test.addOutputs(output2);
 *  // Run.
 *  // Will according to the type and shape of arguments(inputs_/outputs_),
 *  // automatic initialization cpu and gpu function required arguments
 *  // (cpuInputs_/cpuOutputs_/gpuInputs_/gpuOutputs_).
 *  // Call the CPU and GPU Function calculation results.
 *  // Compares CPU and GPU calculation results for consistency.
 *  test.run();
 */
class FunctionCompare {
public:
  FunctionCompare(const std::string& name, const FuncConfig& config)
      : cpu(FunctionBase::funcRegistrar_.createByType(name + "-CPU")),
        gpu(FunctionBase::funcRegistrar_.createByType(name + "-GPU")) {
    cpu->init(config);
    gpu->init(config);
  }

  void addInputs(const BufferArg& input) { inputs.push_back(input); }

  void addOutputs(const BufferArg& output) { outputs.push_back(output); }

  void run() {
    // prepare cpu/gpu arguments
    prepareArgs();

    // function calculate
    cpu->calc(cpuInputs, cpuOutputs);
    gpu->calc(gpuInputs, gpuOutputs);

    // check outputs and inouts
    auto checkArgs = [=](const BufferArgs& cpuArgs, const BufferArgs& gpuArgs) {
      for (size_t i = 0; i < cpuArgs.size(); i++) {
        auto cpu = cpuArgs[i];
        auto gpu = gpuArgs[i];
        CpuVector cpuVector(cpu.shape().getElements(), (real*)cpu.getData());
        GpuVector gpuVector(cpu.shape().getElements(), (real*)gpu.getData());

        autotest::TensorCheckErr(cpuVector, gpuVector);
      }
    };
    checkArgs(cpuOutputs, gpuOutputs);
  }
#if 0
  void cmpWithArg(const Arguments& inputs,
                  const Arguments& outputs,
                  const Arguments& inouts) {
    // init cpu and gpu arguments
    auto initArgs = [=](
        Arguments& cpuArgs, Arguments& gpuArgs, const Arguments& inArgs) {
      for (const auto arg : inArgs) {
        size_t size = sizeof(real);
        for (const auto dim : arg.dims_) {
          size *= dim;
        }
        if (arg.getData()) {
          // todo(tianbing), waste unnecessary mem here
          cpuMemory.emplace_back(std::make_shared<CpuMemoryHandle>(size));
          gpuMemory.emplace_back(std::make_shared<GpuMemoryHandle>(size));
          cpuArgs.emplace_back(Tensor((real*)arg.getData(), arg.dims_));
          gpuArgs.emplace_back(Tensor((real*)arg.getData(), arg.dims_));
          // already init outside
        } else {
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
      }
    };
    initArgs(cpuInputs, gpuInputs, inputs);
    initArgs(cpuOutputs, gpuOutputs, outputs);

    // function calculate
    cpu->calc(cpuInputs, cpuOutputs);
    gpu->calc(gpuInputs, gpuOutputs);

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
  }
#endif

  std::shared_ptr<FunctionBase> getCpuFunction() const { return cpu; }

  std::shared_ptr<FunctionBase> getGpuFunction() const { return gpu; }

protected:
  void prepareArgs() {
    // TODO, if inputs has data
  }

  void createArg(BufferArgs& cpuArgs, BufferArgs& gpuArgs, BufferArg& arg) {
    size_t size = arg.shape().getElements() * sizeOfValuType(arg.valueType());
    cpuMemory_.emplace_back(std::make_shared<CpuMemoryHandle>(size));
    gpuMemory_.emplace_back(std::make_shared<GpuMemoryHandle>(size));

    cpuArgs.emplace_back(
        BufferArg(cpuMemory_.back()->getBuf()), arg.valueType(), arg.shape());
    gpuArgs.emplace_back(
        BufferArg(gpuMemory_.back()->getBuf()), arg.valueType(), arg.shape());
  }

  void createArg(BufferArgs& cpuArgs, BufferArgs& gpuArgs, SequenceArg& arg) {
    size_t batchSize = arg.shape()[0];
    size_t numSeqs = batchSize / 10 + 1;

    size_t sizeId = (numSeqs + 1) * sizeOfValuType(VALUE_TYPE_INT32);
    cpuMemory_.emplace_back(std::make_shared<CpuMemoryHandle>(size));
    gpuMemory_.emplace_back(std::make_shared<GpuMemoryHandle>(size));

    TensorShape seqsId({numSeqs + 1});
    void* cpuBuffer = cpuMemory_.back()->getBuf();
    void* gpuBuffer = gpuMemory_.back()->getBuf();

    size_t size = arg.shape().getElements() * sizeOfValuType(arg.valueType());
    cpuMemory_.emplace_back(std::make_shared<CpuMemoryHandle>(size));
    gpuMemory_.emplace_back(std::make_shared<GpuMemoryHandle>(size));

    cpuArgs.emplace_back(SequenceArg(cpuMemory_.back()->getBuf(),
                                     arg.valueType(),
                                     arg.shape(),
                                     SequenceIdArg(cpuBuffer, seqsId)));
    gpuArgs.emplace_back(SequenceArg(gpuMemory_.back()->getBuf(),
                                     arg.valueType(),
                                     arg.shape(),
                                     SequenceIdArg(gpuBuffer, seqsId)));
  }

  // only init cpu argument, gpu argument copy from cpu argument.
  void initArg(BufferArg& arg) {
    CpuVector vector(arg.shape().getElements(), (real*)arg.data());
    vector.uniform(0.001, 1);
  }

  void initArg(SequenceIdArg& arg, size_t batchSize) {
    size_t numSeqs = arg.numSeqs();
    int* buf = arg.data();
    int pos = 0;
    size_t maxLen = 2 * batchSize / numSeqs;
    for (int i = 0; i < numSeqs; ++i) {
      int len = uniformRandom(
                    std::min<int64_t>(maxLen, batchSize - pos - numSeqs + i)) +
                1;
      buf[i] = pos;
      pos += len;
      VLOG(1) << " len=" << len;
    }
    buf[numSeqs] = batchSize;
  }

protected:
  std::shared_ptr<FunctionBase> cpu;
  std::shared_ptr<FunctionBase> gpu;
  std::vector<CpuMemHandlePtr> cpuMemory_;
  std::vector<GpuMemHandlePtr> gpuMemory_;
  // inputs and outputs
  BufferArgs inputs;
  BufferArgs outputs;
  BufferArgs cpuInputs_;
  BufferArgs cpuOutputs_;
  BufferArgs gpuInputs_;
  BufferArgs gpuOutputs_;
};

}  // namespace paddle
