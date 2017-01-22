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
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"
#include "paddle/math/Vector.h"
#include "paddle/math/tests/TensorCheck.h"
#include "paddle/testing/TestUtil.h"

namespace paddle {

typedef std::shared_ptr<BufferArg> BufferArgPtr;

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
      : cpuFunc_(FunctionBase::funcRegistrar_.createByType(name + "-CPU")),
        gpuFunc_(FunctionBase::funcRegistrar_.createByType(name + "-GPU")) {
    cpuFunc_->init(config);
    gpuFunc_->init(config);
  }

  ~FunctionCompare() {}

  // input need only contains shape, do not contains data.
  void addInputs(const BufferArg& input) {
    size_t size =
        input.shape().getElements() * sizeOfValuType(input.valueType());
    cpuMemory_.emplace_back(std::make_shared<CpuMemoryHandle>(size));
    gpuMemory_.emplace_back(std::make_shared<GpuMemoryHandle>(size));

    cpuInputs_.emplace_back(
        std::make_shared<BufferArg>(cpuMemory_.back()->getBuf(),
                                    input.valueType(),
                                    input.shape(),
                                    UNSPECIFIED,
                                    input.isTransposed()));
    gpuInputs_.emplace_back(
        std::make_shared<BufferArg>(gpuMemory_.back()->getBuf(),
                                    input.valueType(),
                                    input.shape(),
                                    UNSPECIFIED,
                                    input.isTransposed()));
  }

  // output need only contains shape, do not contains data.
  void addOutputs(const BufferArg& output, ArgType argType = ASSIGN_TO) {
    size_t size =
        output.shape().getElements() * sizeOfValuType(output.valueType());
    cpuMemory_.emplace_back(std::make_shared<CpuMemoryHandle>(size));
    gpuMemory_.emplace_back(std::make_shared<GpuMemoryHandle>(size));

    cpuOutputs_.emplace_back(std::make_shared<BufferArg>(
        cpuMemory_.back()->getBuf(),
        output.valueType(),
        output.shape(),
        // todo(tianbing), argType = output.getArgType(), but default ASSIGN_TO
        argType,
        output.isTransposed()));
    gpuOutputs_.emplace_back(std::make_shared<BufferArg>(
        gpuMemory_.back()->getBuf(),
        output.valueType(),
        output.shape(),
        // todo(tianbing), argType = output.getArgType(), but default ASSIGN_TO
        argType,
        output.isTransposed()));
  }

  /// add and init output sparse matrix
  void addOutputs(const SparseMatrixArg& output, ArgType argType = ASSIGN_TO) {
    cpuSparse_ = std::make_shared<CpuSparseMatrix>(output.shape()[0],
                                                   output.shape()[1],
                                                   output.nnz(),
                                                   output.dataType(),
                                                   output.dataFormat(),
                                                   output.isTransposed());

    gpuSparse_ = std::make_shared<GpuSparseMatrix>(output.shape()[0],
                                                   output.shape()[1],
                                                   output.nnz(),
                                                   output.dataType(),
                                                   output.dataFormat(),
                                                   output.isTransposed());

    /// init sparse matrix
    hl_stream_t stream(HPPL_STREAM_1);
    cpuSparse_->randomizeUniform();
    gpuSparse_->copyFrom(*cpuSparse_, stream);
    hl_stream_synchronize(stream);

    cpuOutputs_.emplace_back(
        std::make_shared<SparseMatrixArg>(*cpuSparse_, argType));
    gpuOutputs_.emplace_back(
        std::make_shared<SparseMatrixArg>(*gpuSparse_, argType));
  }

  void addInputs(const SequenceArg& input) {
    size_t batchSize = input.shape()[0];
    size_t numSeqs = batchSize / 10 + 1;

    size_t sizeId = (numSeqs + 1) * sizeOfValuType(VALUE_TYPE_INT32);
    cpuMemory_.emplace_back(std::make_shared<CpuMemoryHandle>(sizeId));
    gpuMemory_.emplace_back(std::make_shared<GpuMemoryHandle>(sizeId));

    TensorShape seqsId({numSeqs + 1});
    // void* cpuBuffer = cpuMemory_.back()->getBuf();
    // void* gpuBuffer = gpuMemory_.back()->getBuf();

    size_t size =
        input.shape().getElements() * sizeOfValuType(input.valueType());
    cpuMemory_.emplace_back(std::make_shared<CpuMemoryHandle>(size));
    gpuMemory_.emplace_back(std::make_shared<GpuMemoryHandle>(size));

    // TODO: need be implemented.
  }

  void addInputs(const SparseMatrixArg& input) {
    cpuSparse_ = std::make_shared<CpuSparseMatrix>(input.shape()[0],
                                                   input.shape()[1],
                                                   input.nnz(),
                                                   input.dataType(),
                                                   input.dataFormat(),
                                                   input.isTransposed());

    gpuSparse_ = std::make_shared<GpuSparseMatrix>(input.shape()[0],
                                                   input.shape()[1],
                                                   input.nnz(),
                                                   input.dataType(),
                                                   input.dataFormat(),
                                                   input.isTransposed());

    /// init sparse matrix
    hl_stream_t stream(HPPL_STREAM_1);
    cpuSparse_->randomizeUniform();
    gpuSparse_->copyFrom(*cpuSparse_, stream);
    hl_stream_synchronize(stream);

    cpuInputs_.emplace_back(std::make_shared<SparseMatrixArg>(*cpuSparse_));
    gpuInputs_.emplace_back(std::make_shared<SparseMatrixArg>(*gpuSparse_));
  }

  void run() {
    // prepare cpu/gpu arguments
    initInputs();

    initOutputs();
    // function calculate
    auto callFunction = [](FunctionBase* function,
                           std::vector<BufferArgPtr>& inputs,
                           std::vector<BufferArgPtr>& outputs) {
      BufferArgs inArgs;
      BufferArgs outArgs;
      for (auto arg : inputs) {
        inArgs.addArg(*arg);
      }
      for (auto arg : outputs) {
        outArgs.addArg(*arg);
      }
      function->calc(inArgs, outArgs);
    };

    callFunction(cpuFunc_.get(), cpuInputs_, cpuOutputs_);
    callFunction(gpuFunc_.get(), gpuInputs_, gpuOutputs_);

    // check outputs
    compareOutputs();
  }

  std::shared_ptr<FunctionBase> getCpuFunction() const { return cpuFunc_; }

  std::shared_ptr<FunctionBase> getGpuFunction() const { return gpuFunc_; }

protected:
  void initInputs() {
    for (size_t i = 0; i < cpuInputs_.size(); i++) {
      if (cpuInputs_[i]->isSparseArg()) {
        continue;  /// sparse matrix already init
      }

      initArg(*cpuInputs_[i]);

      // TODO: Need a BufferCopy used to copy from one BufferArg to another.
      CpuVector cpuVector(cpuInputs_[i]->shape().getElements(),
                          (real*)cpuInputs_[i]->data());
      GpuVector gpuVector(gpuInputs_[i]->shape().getElements(),
                          (real*)gpuInputs_[i]->data());

      gpuVector.copyFrom(cpuVector);
    }
  }

  void initOutputs() {
    for (size_t i = 0; i < cpuOutputs_.size(); i++) {
      if (cpuOutputs_[i]->isSparseArg()) {
        continue;  /// sparse matrix already init
      }

      initArg(*cpuOutputs_[i]);

      // TODO: Need a BufferCopy used to copy from one BufferArg to another.
      CpuVector cpuVector(cpuOutputs_[i]->shape().getElements(),
                          (real*)cpuOutputs_[i]->data());
      GpuVector gpuVector(gpuOutputs_[i]->shape().getElements(),
                          (real*)gpuOutputs_[i]->data());

      gpuVector.copyFrom(cpuVector);
    }
  }

  void compareOutputs() {
    for (size_t i = 0; i < cpuOutputs_.size(); i++) {
      // TODO, Need a BufferCheck used to compare the two buffers.
      const auto cpu = cpuOutputs_[i];
      const auto gpu = gpuOutputs_[i];
      CHECK_EQ(cpu->numElements(), gpu->numElements());
      CpuVector cpuVector(cpu->numElements(), (real*)cpu->data());
      GpuVector gpuVector(gpu->numElements(), (real*)gpu->data());
      autotest::TensorCheckErr(cpuVector, gpuVector);
    }
  }

  // only init cpu argument, gpu argument copy from cpu argument.
  void initArg(BufferArg& arg) {
    CpuVector vector(arg.shape().getElements(), (real*)arg.data());
    vector.uniform(0.001, 1);
  }

  void initArg(SequenceIdArg& arg, size_t batchSize) {
    size_t numSeqs = arg.numSeqs();
    int* buf = reinterpret_cast<int*>(arg.data());
    int pos = 0;
    size_t maxLen = 2 * batchSize / numSeqs;
    for (int i = 0; i < (int)numSeqs; ++i) {
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
  std::shared_ptr<FunctionBase> cpuFunc_;
  std::shared_ptr<FunctionBase> gpuFunc_;
  std::vector<CpuMemHandlePtr> cpuMemory_;
  std::vector<GpuMemHandlePtr> gpuMemory_;
  std::vector<BufferArgPtr> cpuInputs_;
  std::vector<BufferArgPtr> cpuOutputs_;
  std::vector<BufferArgPtr> gpuInputs_;
  std::vector<BufferArgPtr> gpuOutputs_;
  std::shared_ptr<CpuSparseMatrix> cpuSparse_;
  std::shared_ptr<GpuSparseMatrix> gpuSparse_;
};

}  // namespace paddle
