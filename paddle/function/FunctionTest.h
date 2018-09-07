/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/math/tests/TensorCheck.h"
#include "paddle/testing/TestUtil.h"

namespace paddle {

typedef std::shared_ptr<BufferArg> BufferArgPtr;

namespace test {
template <DeviceType DType>
struct Allocator;

template <>
struct Allocator<DEVICE_TYPE_CPU> {
  using type = CpuMemoryHandle;
};

template <>
struct Allocator<DEVICE_TYPE_GPU> {
  using type = GpuMemoryHandle;
};

// Copy argument1 to argument2
template <DeviceType DType1, DeviceType DType2>
class CopyArgument {
 public:
  void operator()(const BufferArg& arg1, BufferArg& arg2) {
    CHECK_EQ(arg1.valueType(), arg2.valueType());
    CHECK_LE(arg1.shape().getElements(), arg2.shape().getElements());

    if (arg1.valueType() == VALUE_TYPE_INT32) {
      IVectorPtr vector1 =
          IVector::create((int*)arg1.data(),
                          arg1.shape().getElements(),
                          DType1 == DEVICE_TYPE_CPU ? false : true);
      IVectorPtr vector2 =
          IVector::create((int*)arg2.data(),
                          arg2.shape().getElements(),
                          DType2 == DEVICE_TYPE_CPU ? false : true);
      vector2->copyFrom(*vector1);
    } else {
      VectorPtr vector1 =
          Vector::create((real*)arg1.data(),
                         arg1.shape().getElements(),
                         DType1 == DEVICE_TYPE_CPU ? false : true);
      VectorPtr vector2 =
          Vector::create((real*)arg2.data(),
                         arg2.shape().getElements(),
                         DType2 == DEVICE_TYPE_CPU ? false : true);
      vector2->copyFrom(*vector1);
    }
  }
};
}  // namespace test

/**
 * \brief A class for comparing two Functions of different implementations.
 *        For example, can be used to compare the CPU and GPU implementation
 *        of the function is consistent.
 *
 * Use case:
 *  // Initializes a test object, the corresponding cpu and gpu Function
 *  // are constructed according to FunctionName and FuncConfig.
 *  CpuGpuFuncCompare test(FunctionName, FuncConfig);
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
template <DeviceType DType1, DeviceType DType2>
class Compare2Function {
 public:
  typedef typename test::Allocator<DType1>::type Allocator1;
  typedef typename test::Allocator<DType2>::type Allocator2;
  typedef typename Tensor<real, DType1>::Vector Vector1;
  typedef typename Tensor<real, DType2>::Vector Vector2;
  typedef typename Tensor<real, DType1>::SparseMatrix SparseMatrix1;
  typedef typename Tensor<real, DType2>::SparseMatrix SparseMatrix2;

  Compare2Function(const std::string& name1,
                   const std::string& name2,
                   const FuncConfig& config)
      : function1_(FunctionBase::funcRegistrar_.createByType(name1)),
        function2_(FunctionBase::funcRegistrar_.createByType(name2)) {
    function1_->init(config);
    function2_->init(config);
    initArgsCallback_ = nullptr;
  }

  ~Compare2Function() {}

  // input need only contains shape, do not contains data.
  void addInputs(const BufferArg& input) {
    size_t size =
        input.shape().getElements() * sizeOfValuType(input.valueType());
    func1Memory_.emplace_back(std::make_shared<Allocator1>(size));
    func2Memory_.emplace_back(std::make_shared<Allocator2>(size));

    func1Inputs_.emplace_back(std::make_shared<BufferArg>(
        func1Memory_.back()->getBuf(), input.valueType(), input.shape()));
    func2Inputs_.emplace_back(std::make_shared<BufferArg>(
        func2Memory_.back()->getBuf(), input.valueType(), input.shape()));
  }

  // assume one copy of sequence is shared by different SequenceArgs
  void addSequence(const SequenceIdArg& input) {
    CHECK_EQ(input.shape().ndims(), 1UL);
    size_t batchSize = input.shape()[0];
    size_t numSeqs = batchSize / 10 + 1;
    size_t sizeId = (numSeqs + 1) * sizeOfValuType(VALUE_TYPE_INT32);
    func1Memory_.emplace_back(std::make_shared<Allocator1>(sizeId));
    func2Memory_.emplace_back(std::make_shared<Allocator2>(sizeId));
    seq1_ = std::make_shared<SequenceIdArg>(func1Memory_.back()->getBuf(),
                                            TensorShape{numSeqs + 1});
    seq2_ = std::make_shared<SequenceIdArg>(func2Memory_.back()->getBuf(),
                                            TensorShape{numSeqs + 1});
    /// init sequence Id
    initArg(*seq1_, batchSize);

    copyArg_(*seq1_, *seq2_);
  }

  void addInputs(const SequenceArg& input) {
    CHECK_EQ(input.shape().ndims(), 2UL);
    size_t batchSize = input.shape()[0];
    if (!seq1_ || !seq2_) {  // sequence not exist
      addSequence(SequenceIdArg(TensorShape{batchSize}));
    }

    size_t size =
        input.shape().getElements() * sizeOfValuType(input.valueType());
    func1Memory_.emplace_back(std::make_shared<Allocator1>(size));
    func2Memory_.emplace_back(std::make_shared<Allocator2>(size));

    /// SequenceArg
    func1Inputs_.emplace_back(
        std::make_shared<SequenceArg>(func1Memory_.back()->getBuf(),
                                      input.valueType(),
                                      input.shape(),
                                      *seq1_));
    func2Inputs_.emplace_back(
        std::make_shared<SequenceArg>(func2Memory_.back()->getBuf(),
                                      input.valueType(),
                                      input.shape(),
                                      *seq2_));
  }

  void registerInitCallback(std::function<void(BufferArg&, size_t)> callback) {
    initArgsCallback_ = callback;
  }

  // output need only contains shape, do not contains data.
  void addOutputs(const BufferArg& output, ArgType argType = ASSIGN_TO) {
    size_t size =
        output.shape().getElements() * sizeOfValuType(output.valueType());
    func1Memory_.emplace_back(std::make_shared<Allocator1>(size));
    func2Memory_.emplace_back(std::make_shared<Allocator2>(size));

    func1Outputs_.emplace_back(
        std::make_shared<BufferArg>(func1Memory_.back()->getBuf(),
                                    output.valueType(),
                                    output.shape(),
                                    argType));
    func2Outputs_.emplace_back(
        std::make_shared<BufferArg>(func2Memory_.back()->getBuf(),
                                    output.valueType(),
                                    output.shape(),
                                    argType));
  }

  /// add and init output sparse matrix
  void addOutputs(const SparseMatrixArg& output, ArgType argType = ASSIGN_TO) {
    sparse1_ = std::make_shared<SparseMatrix1>(
        output.shape()[0],
        output.shape()[1],
        output.nnz(),
        static_cast<SparseValueType>(output.dataType()),
        static_cast<SparseFormat>(output.dataFormat()));

    sparse2_ = std::make_shared<SparseMatrix2>(
        output.shape()[0],
        output.shape()[1],
        output.nnz(),
        static_cast<SparseValueType>(output.dataType()),
        static_cast<SparseFormat>(output.dataFormat()));

    /// init sparse matrix
    hl_stream_t stream(HPPL_STREAM_1);
    sparse1_->randomizeUniform();
    sparse2_->copyFrom(*sparse1_, stream);
    hl_stream_synchronize(stream);

    func1Outputs_.emplace_back(
        std::make_shared<SparseMatrixArg>(*sparse1_, argType));
    func2Outputs_.emplace_back(
        std::make_shared<SparseMatrixArg>(*sparse2_, argType));
  }

  void addOutputs(const SequenceArg& output, ArgType argType = ASSIGN_TO) {
    CHECK_EQ(output.shape().ndims(), 2UL);
    size_t batchSize = output.shape()[0];

    if (!seq1_ || !seq2_) {  // sequence not exist
      addSequence(SequenceIdArg(TensorShape{batchSize}));
    }
    size_t size =
        output.shape().getElements() * sizeOfValuType(output.valueType());
    func1Memory_.emplace_back(std::make_shared<Allocator1>(size));
    func2Memory_.emplace_back(std::make_shared<Allocator2>(size));

    /// SequenceArg
    func1Outputs_.emplace_back(
        std::make_shared<SequenceArg>(func1Memory_.back()->getBuf(),
                                      output.valueType(),
                                      output.shape(),
                                      *seq1_,
                                      argType));
    func2Outputs_.emplace_back(
        std::make_shared<SequenceArg>(func2Memory_.back()->getBuf(),
                                      output.valueType(),
                                      output.shape(),
                                      *seq2_,
                                      argType));
  }

  void addInputs(const SparseMatrixArg& input) {
    sparse1_ = std::make_shared<SparseMatrix1>(
        input.shape()[0],
        input.shape()[1],
        input.nnz(),
        static_cast<SparseValueType>(input.dataType()),
        static_cast<SparseFormat>(input.dataFormat()));

    sparse2_ = std::make_shared<SparseMatrix2>(
        input.shape()[0],
        input.shape()[1],
        input.nnz(),
        static_cast<SparseValueType>(input.dataType()),
        static_cast<SparseFormat>(input.dataFormat()));

    /// init sparse matrix
    hl_stream_t stream(HPPL_STREAM_1);
    sparse1_->randomizeUniform();
    sparse2_->copyFrom(*sparse1_, stream);
    hl_stream_synchronize(stream);

    func1Inputs_.emplace_back(std::make_shared<SparseMatrixArg>(*sparse1_));
    func2Inputs_.emplace_back(std::make_shared<SparseMatrixArg>(*sparse2_));
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

    callFunction(function1_.get(), func1Inputs_, func1Outputs_);
    callFunction(function2_.get(), func2Inputs_, func2Outputs_);

    // check outputs
    compareOutputs();
  }

  std::shared_ptr<FunctionBase> getFunction1() const { return function1_; }

  std::shared_ptr<FunctionBase> getFunction2() const { return function2_; }

 protected:
  // only init cpu argument, gpu argument copy from cpu argument.
  void initArg(BufferArg& arg) {
    Vector1 vector(arg.shape().getElements(), (real*)arg.data());
    vector.uniform(0.001, 1);
  }

  void initArg(SequenceArg& arg) {
    /// init only matrix
    Vector1 vector(arg.shape().getElements(), (real*)arg.data());
    vector.uniform(0.001, 1);
  }

  void initArg(SequenceIdArg& arg, size_t batchSize) {
    size_t numSeqs = arg.numSeqs();
    int* buf = reinterpret_cast<int*>(arg.data());
    int pos = 0;
    size_t maxLen = 2 * batchSize / numSeqs;
    for (int i = 0; i < (int)numSeqs; ++i) {
      int len = 1 + uniformRandom(std::min<int64_t>(
                        maxLen, batchSize - pos - numSeqs + i));
      buf[i] = pos;
      pos += len;
      VLOG(1) << " len=" << len;
    }
    buf[numSeqs] = batchSize;
  }

  void initInputs() {
    for (size_t i = 0; i < func1Inputs_.size(); i++) {
      if (func1Inputs_[i]->isSparseArg()) {
        continue;  /// sparse matrix already init
      }

      if (func1Inputs_[i]->isSequenceArg()) {
        initArg(dynamic_cast<SequenceArg&>(*func1Inputs_[i]));
      } else {
        initArg(*func1Inputs_[i]);
      }

      if (initArgsCallback_ != nullptr) {
        initArgsCallback_(*func1Inputs_[i], i);
      }

      copyArg_(*func1Inputs_[i], *func2Inputs_[i]);
    }
  }

  void initOutputs() {
    for (size_t i = 0; i < func1Outputs_.size(); i++) {
      if (func1Outputs_[i]->isSparseArg()) {
        continue;  /// sparse matrix already init
      }

      if (func1Outputs_[i]->isSequenceArg()) {
        initArg(dynamic_cast<SequenceArg&>(*func1Outputs_[i]));
      } else {
        initArg(*func1Outputs_[i]);
      }

      copyArg_(*func1Outputs_[i], *func2Outputs_[i]);
    }
  }

  void compareOutputs() {
    for (size_t i = 0; i < func1Outputs_.size(); i++) {
      // TODO, Need a BufferCheck used to compare the two buffers.
      const auto cpu = func1Outputs_[i];
      const auto gpu = func2Outputs_[i];
      CHECK_EQ(cpu->numElements(), gpu->numElements());
      Vector1 cpuVector(cpu->numElements(), (real*)cpu->data());
      Vector2 gpuVector(gpu->numElements(), (real*)gpu->data());
      autotest::TensorCheckErr(cpuVector, gpuVector);
    }
  }

 protected:
  std::shared_ptr<FunctionBase> function1_;
  std::shared_ptr<FunctionBase> function2_;
  std::vector<std::shared_ptr<Allocator1>> func1Memory_;
  std::vector<std::shared_ptr<Allocator2>> func2Memory_;
  std::vector<BufferArgPtr> func1Inputs_;
  std::vector<BufferArgPtr> func1Outputs_;
  std::vector<BufferArgPtr> func2Inputs_;
  std::vector<BufferArgPtr> func2Outputs_;
  std::shared_ptr<SparseMatrix1> sparse1_;
  std::shared_ptr<SparseMatrix2> sparse2_;
  std::shared_ptr<SequenceIdArg> seq1_;
  std::shared_ptr<SequenceIdArg> seq2_;
  test::CopyArgument<DType1, DType2> copyArg_;
  std::function<void(BufferArg&, size_t)> initArgsCallback_;
};

class CpuGpuFuncCompare
    : public Compare2Function<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU> {
 public:
  CpuGpuFuncCompare(const std::string& name, const FuncConfig& config)
      : Compare2Function(name + "-CPU", name + "-GPU", config) {}

  ~CpuGpuFuncCompare() {}
};

}  // namespace paddle
