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

#include "RowConvOp.h"
#include <iostream>
#include "paddle/math/Vector.h"

namespace paddle {

template <>
void RowConv<DEVICE_TYPE_CPU>(CpuMatrix& out,
                              const CpuMatrix& in,
                              const CpuMatrix& filter,
                              const CpuIVector& seq) {
  const int* starts = seq.getData();
  const size_t numSeq = seq.getSize() - 1;
  const size_t contextLength = filter.getHeight();
  for (size_t i = 0; i < numSeq; ++i) {
    size_t begin = starts[i];
    size_t end = starts[i + 1];
    for (size_t j = begin; j < end; ++j) {
      MatrixPtr x;
      MatrixPtr w;
      if ((j + contextLength) < end) {
        x = (const_cast<CpuMatrix&>(in)).subMatrix(j, contextLength);
        w = (const_cast<CpuMatrix&>(filter)).subMatrix(0, contextLength);
      } else {
        x = (const_cast<CpuMatrix&>(in)).subMatrix(j, end - j);
        w = (const_cast<CpuMatrix&>(filter)).subMatrix(0, end - j);
      }
      MatrixPtr y = out.subMatrix(j, 1);
      y->addDotMulVMM(*x, *w);
    }
  }
}

template <>
void RowConvGrad<DEVICE_TYPE_CPU>(const CpuMatrix& outG,
                                  const CpuMatrix& in,
                                  const CpuMatrix& filter,
                                  CpuMatrix& inG,
                                  CpuMatrix& filterG,
                                  const CpuIVector& seq) {
  // gradient w.r.t filter
  const int* starts = seq.getData();
  const size_t numSeq = seq.getSize() - 1;
  const size_t contextLength = filter.getHeight();
  if (filterG) {
    for (size_t i = 0; i < numSeq; ++i) {
      size_t begin = starts[i];
      size_t end = starts[i + 1];
      size_t steps = end - begin;
      for (size_t j = 0; j < contextLength; ++j) {
        MatrixPtr x =
            (const_cast<CpuMatrix&>(in)).subMatrix(begin + j, steps - j);
        MatrixPtr dy =
            (const_cast<CpuMatrix&>(outG)).subMatrix(begin, steps - j);
        MatrixPtr dw = filterG.subMatrix(j, 1);
        dw->addDotMulVMM(*dy, *x);
      }
    }
  }

  // gradient w.r.t input feature
  if (inG) {
    for (size_t i = 0; i < numSeq; ++i) {
      size_t begin = starts[i];
      size_t end = starts[i + 1];
      size_t steps = end - begin;
      for (size_t j = 0; j < steps; ++j) {
        MatrixPtr dx = inG.subMatrix(begin + j, 1);
        for (size_t t = 0; t < contextLength; ++t) {
          if ((int(j) - int(t)) >= 0) {
            MatrixPtr dy =
                (const_cast<CpuMatrix&>(outG)).subMatrix(begin + j - t, 1);
            MatrixPtr w = (const_cast<CpuMatrix&>(filter)).subMatrix(t, 1);
            dx->addDotMul(*dy, *w, 1.0, 1.0);
          }
        }
      }
    }
  }
}

/**
 * \brief TODO(qingqing)
 *
 */

template <DeviceType Device>
class RowConvFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {}

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    // check
    CHECK_EQ(2UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);
    CHECK(inputs[0].isSequenceArg() && outputs[0].isSequenceArg())
        << "SequenceArg required here.";
    const auto in = dynamic_cast<const SequenceArg&>(inputs[0]);
    auto out = dynamic_cast<const SequenceArg&>(outputs[0]);
    auto w = inputs[1];
    CHECK(in.data() && out.data() && in.getSequenceId().data());
    CHECK_EQ(in.shape().ndims(), 2UL);
    CHECK_EQ(out.shape().ndims(), 2UL);
    CHECK_EQ(in.shape()[1], out.shape()[1]);
    CHECK_EQ(in.shape()[0], out.shape()[0]);
    CHECK_EQ(w.shape()[1], in.shape()[1]);

    auto outMat = out.matrix<Device>();
    const auto inMat = in.matrix<Device>();
    const auto wMat = w.matrix<Device>();
    const auto seqId = in.getSequenceId().vector<int, Device>();

    RowConv<Device>(outMat, inMat, wMat, seqId);
  }
};
/**
 * \brief TODO(qingqing)
 *
 * Argument in this Function:
 */

template <DeviceType Device>
class RowConvGradFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {}

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    const auto outGrad = dynamic_cast<const SequenceArg&>(inputs[0]);
    const auto in = dynamic_cast<const SequenceArg&>(inputs[1]);
    const auto w = inputs[2];
    auto inGrad = dynamic_cast<const SequenceArg&>(outputs[0]);
    auto wGrad = outputs[1];

    const auto outGMat = outGrad.matrix<Device>();
    const auto inMat = in.matrix<Device>();
    const auto wMat = w.matrix<Device>();
    auto inGMat = inGrad.data()
                      ? inGrad.matrix<Device>()
                      : typename Tensor<real, Device>::Matrix(nullptr, 0, 0);
    auto wGMat = wGrad.data()
                     ? wGrad.matrix<Device>()
                     : typename Tensor<real, Device>::Matrix(nullptr, 0, 0);
    const auto seqId = in.getSequenceId().vector<int, Device>();

    std::cout << "in:" << std::endl;
    for (int i = 0; i < inMat.getHeight(); ++i) {
      for (int j = 0; j < inMat.getWidth(); ++j) {
        std::cout << outGMat.getElement(i, j) << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "w:" << std::endl;
    for (int i = 0; i < wMat.getHeight(); ++i) {
      for (int j = 0; j < wMat.getWidth(); ++j) {
        std::cout << wMat.getElement(i, j) << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "w:" << std::endl;
    for (int i = 0; i < seqId.getSize(); ++i) {
      std::cout << seqId.getElement(i) << " ";
    }
    std::cout << std::endl;

    RowConvGrad<Device>(outGMat, inMat, wMat, inGMat, wGMat, seqId);

    std::cout << std::endl << "out:" << std::endl;
    for (int i = 0; i < inGMat.getHeight(); ++i) {
      for (int j = 0; j < inGMat.getWidth(); ++j) {
        std::cout << inGMat.getElement(i, j) << " ";
      }
      std::cout << std::endl;
    }
  }
};

REGISTER_TYPED_FUNC(RowConv, CPU, RowConvFunc);
REGISTER_TYPED_FUNC(RowConvGrad, CPU, RowConvGradFunc);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(RowConv, GPU, RowConvFunc);
REGISTER_TYPED_FUNC(RowConvGrad, GPU, RowConvGradFunc);
#endif

}  // namespace paddle
